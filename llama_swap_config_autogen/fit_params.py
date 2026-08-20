"""VRAM estimation via llama.cpp's native `fit-params --fit-print on`."""

import hashlib
import json
import logging
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CACHE_PATH = Path.home() / ".cache" / "llama_swap_config_autogen" / "fit_params_vram.json"
CACHE_SCHEMA_VERSION = 1


class FitParamsCache(BaseModel):
    version: int = CACHE_SCHEMA_VERSION
    entries: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def load(cls) -> "FitParamsCache":
        if CACHE_PATH.exists():
            try:
                data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                cache = cls.model_validate(data)
                if cache.version != CACHE_SCHEMA_VERSION:
                    logger.info(
                        "Discarding fit-params VRAM cache due to schema version change (%s -> %s)",
                        cache.version,
                        CACHE_SCHEMA_VERSION,
                    )
                    return cls()
                return cache
            except Exception as e:
                logger.warning("Failed to load fit-params VRAM cache, starting fresh: %s", e)
        return cls()

    def save(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def get(self, key: str) -> float | None:
        return self.entries.get(key)

    def set(self, key: str, vram_gib: float) -> None:
        self.entries[key] = vram_gib


def apply_path_prefix_map(path: Path, path_prefix_map: dict[str, str]) -> str:
    """Rewrite a host-side path to the runtime-visible path using the longest matching prefix."""
    path_str = str(path)
    best_prefix = ""
    for host_prefix in path_prefix_map:
        if path_str.startswith(host_prefix) and len(host_prefix) > len(best_prefix):
            best_prefix = host_prefix
    if not best_prefix:
        return path_str
    return path_prefix_map[best_prefix] + path_str[len(best_prefix) :]


def build_cache_key(
    path_model: Path,
    ngl: int,
    ctx: int,
    extra_args: tuple[str, ...],
) -> str:
    stat = path_model.stat()
    payload = json.dumps(
        {
            "path": str(path_model),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "ngl": ngl,
            "ctx": ctx,
            "extra_args": extra_args,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_fit_print_output(output: str) -> float | None:
    """Sum the model+context+compute MiB across GPU device rows from `--fit-print on` stdout.

    Rows look like `<device> <model_mib> <context_mib> <compute_mib>`, one per device plus a
    trailing `Host ...` row. Only non-Host rows are summed, since we want GPU VRAM specifically.
    """
    total_mib = 0
    found = False
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        device, model_mib, context_mib, compute_mib = parts
        if device == "Host":
            continue
        try:
            total_mib += int(model_mib) + int(context_mib) + int(compute_mib)
            found = True
        except ValueError:
            continue
    if not found:
        return None
    return total_mib / 1024


def estimate_vram_gib_via_fit_params(
    llama_bin: list[str],
    path_model: Path,
    ngl: int,
    ctx: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
    extra_args: list[str],
    path_prefix_map: dict[str, str],
    cache: FitParamsCache,
    extra_gpu_gib: float = 0.0,
) -> float | None:
    """Return the GPU VRAM estimate in GiB using llama.cpp's native fit-params tool, or None on failure."""
    cache_key = build_cache_key(path_model, ngl, ctx, tuple(extra_args))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached + extra_gpu_gib

    runtime_path = apply_path_prefix_map(path_model, path_prefix_map)
    command = [
        *llama_bin,
        "fit-params",
        "--model",
        runtime_path,
        "-c",
        str(ctx),
        "-ngl",
        str(ngl),
        "--fit-print",
        "on",
    ]
    if cache_type_k:
        command += ["--cache-type-k", cache_type_k]
    if cache_type_v:
        command += ["--cache-type-v", cache_type_v]
    command += extra_args

    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except Exception as e:
        logger.warning("fit-params invocation failed for %s: %s", path_model.name, e)
        return None

    vram_gib = parse_fit_print_output(result.stdout)
    if vram_gib is None:
        logger.warning("Could not parse fit-params output for %s: %r", path_model.name, result.stdout)
        return None

    cache.set(cache_key, vram_gib)
    return vram_gib + extra_gpu_gib
