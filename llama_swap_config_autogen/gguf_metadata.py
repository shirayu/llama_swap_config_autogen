"""GGUF metadata reader with file-based cache."""

import json
import logging
from pathlib import Path
from typing import SupportsInt

from gguf import GGUFReader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CACHE_PATH = Path.home() / ".cache" / "llama_swap_config_autogen" / "gguf_metadata.json"
ARCH_FALLBACKS = ["llama", "mistral", "phi3", "gemma", "qwen2"]
CACHE_SCHEMA_VERSION = 3


class GGUFMetadata(BaseModel):
    mtime: float
    size: int
    num_layers: int
    num_heads: int
    num_heads_kv: int
    head_dim: int
    context_length: int
    embedding_length: int
    expert_count: int = 0
    expert_used_count: int = 0
    feed_forward_length: int = 0
    expert_feed_forward_length: int = 0
    expert_shared_feed_forward_length: int = 0


class GGUFMetadataCache(BaseModel):
    version: int = CACHE_SCHEMA_VERSION
    entries: dict[str, GGUFMetadata] = Field(default_factory=dict)

    @classmethod
    def load(cls) -> "GGUFMetadataCache":
        if CACHE_PATH.exists():
            try:
                data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                cache = cls.model_validate(data)
                if cache.version != CACHE_SCHEMA_VERSION:
                    logger.info(
                        "Discarding GGUF metadata cache due to schema version change (%s -> %s)",
                        cache.version,
                        CACHE_SCHEMA_VERSION,
                    )
                    return cls()
                logger.debug("Loaded GGUF metadata cache from %s (%d entries)", CACHE_PATH, len(cache.entries))
                return cache
            except Exception as e:
                logger.warning("Failed to load GGUF metadata cache, starting fresh: %s", e)
        return cls()

    def save(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Saved GGUF metadata cache to %s (%d entries)", CACHE_PATH, len(self.entries))

    def get(self, path: Path) -> GGUFMetadata | None:
        key = str(path)
        entry = self.entries.get(key)
        if entry is None:
            return None
        stat = path.stat()
        if entry.mtime != stat.st_mtime or entry.size != stat.st_size:
            logger.debug("Cache miss (file changed): %s", path)
            del self.entries[key]
            return None
        logger.debug("Cache hit: %s", path)
        return entry

    def set(self, path: Path, metadata: GGUFMetadata) -> None:
        self.entries[str(path)] = metadata


def _read_gguf_metadata(path: Path) -> GGUFMetadata:
    """Read metadata directly from a GGUF file."""
    logger.info("Reading GGUF metadata: %s", path)
    stat = path.stat()

    reader = GGUFReader(str(path), "r")
    kv = {field.name: field for field in reader.fields.values()}

    def coerce_int(value: object, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return coerce_int(value[0], default)
        if isinstance(value, SupportsInt):
            return int(value)
        return default

    def get_int(key: str, default: int = 0) -> int:
        field = kv.get(key)
        if field is None:
            return default
        try:
            return coerce_int(field.contents(0), default)
        except Exception:
            logger.debug("Failed to coerce GGUF field %s to int", key, exc_info=True)
            return default

    def discover_arch_prefixes() -> list[str]:
        prefixes: dict[str, set[str]] = {}
        for key in kv:
            if "." not in key:
                continue
            prefix, suffix = key.split(".", 1)
            prefixes.setdefault(prefix, set()).add(suffix)

        required = {"block_count", "embedding_length", "context_length"}
        bonus = {"attention.head_count", "attention.head_count_kv"}

        ranked = sorted(
            prefixes.items(),
            key=lambda item: (
                required.issubset(item[1]),
                len(item[1].intersection(required | bonus)),
                item[0] in ARCH_FALLBACKS,
            ),
            reverse=True,
        )
        discovered = [prefix for prefix, suffixes in ranked if required.issubset(suffixes)]
        return discovered + [prefix for prefix in ARCH_FALLBACKS if prefix not in discovered]

    archs = discover_arch_prefixes()
    logger.debug("Detected GGUF architecture candidates for %s: %s", path.name, archs)
    num_layers = next((get_int(f"{a}.block_count") for a in archs if get_int(f"{a}.block_count")), 0)
    num_heads = next((get_int(f"{a}.attention.head_count") for a in archs if get_int(f"{a}.attention.head_count")), 0)
    num_heads_kv = next(
        (get_int(f"{a}.attention.head_count_kv") for a in archs if get_int(f"{a}.attention.head_count_kv")),
        num_heads,
    )
    embedding_length = next((get_int(f"{a}.embedding_length") for a in archs if get_int(f"{a}.embedding_length")), 0)
    context_length = next((get_int(f"{a}.context_length") for a in archs if get_int(f"{a}.context_length")), 0)
    expert_count = next((get_int(f"{a}.expert_count") for a in archs if get_int(f"{a}.expert_count")), 0)
    expert_used_count = next((get_int(f"{a}.expert_used_count") for a in archs if get_int(f"{a}.expert_used_count")), 0)
    feed_forward_length = next(
        (get_int(f"{a}.feed_forward_length") for a in archs if get_int(f"{a}.feed_forward_length")),
        0,
    )
    expert_feed_forward_length = next(
        (get_int(f"{a}.expert_feed_forward_length") for a in archs if get_int(f"{a}.expert_feed_forward_length")),
        0,
    )
    expert_shared_feed_forward_length = next(
        (
            get_int(f"{a}.expert_shared_feed_forward_length")
            for a in archs
            if get_int(f"{a}.expert_shared_feed_forward_length")
        ),
        0,
    )

    head_dim = (embedding_length // num_heads) if num_heads > 0 else 0

    metadata = GGUFMetadata(
        mtime=stat.st_mtime,
        size=stat.st_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        context_length=context_length,
        embedding_length=embedding_length,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        feed_forward_length=feed_forward_length,
        expert_feed_forward_length=expert_feed_forward_length,
        expert_shared_feed_forward_length=expert_shared_feed_forward_length,
    )

    logger.debug(
        "  layers=%d, heads=%d, heads_kv=%d, head_dim=%d, ctx=%d, emb=%d, experts=%d, used=%d",
        num_layers,
        num_heads,
        num_heads_kv,
        head_dim,
        context_length,
        embedding_length,
        expert_count,
        expert_used_count,
    )
    if num_layers == 0 or embedding_length == 0 or context_length == 0:
        related_suffixes = (
            "block_count",
            "attention.head_count",
            "attention.head_count_kv",
            "embedding_length",
            "context_length",
        )
        related_keys = [key for key in kv if key.endswith(related_suffixes)]
        logger.debug("Incomplete GGUF metadata for %s", path.name)
        logger.debug("Available related GGUF keys: %s", related_keys)
    return metadata


def get_gguf_metadata(path: Path, cache: GGUFMetadataCache) -> GGUFMetadata:
    """Get GGUF metadata, using cache when available."""
    metadata = cache.get(path)
    if metadata is not None:
        return metadata

    metadata = _read_gguf_metadata(path)
    cache.set(path, metadata)
    return metadata


def estimate_vram_gb(
    metadata: GGUFMetadata,
    file_size_bytes: int,
    ngl: int,
    context_length: int,
    k_cache_bytes: float = 2.0,
    v_cache_bytes: float = 2.0,
    extra_gpu_bytes: int = 0,
) -> float:
    """Estimate VRAM usage in GB.

    Args:
        metadata: GGUF metadata
        file_size_bytes: actual file size in bytes
        ngl: number of GPU layers (-ngl value)
        context_length: context length (-c value)
        k_cache_bytes: bytes per element for K cache
        v_cache_bytes: bytes per element for V cache
        extra_gpu_bytes: extra bytes always placed on GPU, such as mmproj
    """
    num_layers = metadata.num_layers
    if num_layers == 0:
        logger.warning("num_layers=0, falling back to file size only")
        return (file_size_bytes + extra_gpu_bytes) / 1024**3

    # Clamp ngl to [0, num_layers]
    effective_ngl = max(0, min(ngl, num_layers))
    gpu_ratio = effective_ngl / num_layers

    # Model weights on GPU
    model_vram = file_size_bytes * gpu_ratio

    # KV cache on GPU (all layers when using GPU)
    # KV = 2 (K+V) * layers * num_heads_kv * head_dim * context * bytes_per_element
    kv_vram = 0.0
    if effective_ngl > 0 and metadata.num_heads_kv > 0 and metadata.head_dim > 0:
        kv_vram = (
            effective_ngl * metadata.num_heads_kv * metadata.head_dim * context_length * (k_cache_bytes + v_cache_bytes)
        )

    total_bytes = model_vram + kv_vram + extra_gpu_bytes
    total_gb = total_bytes / 1024**3

    logger.debug(
        "VRAM estimate: model=%.2f GB, kv=%.2f GB, total=%.2f GB (ngl=%d/%d, ctx=%d)",
        model_vram / 1024**3,
        kv_vram / 1024**3,
        total_gb,
        effective_ngl,
        num_layers,
        context_length,
    )
    return total_gb
