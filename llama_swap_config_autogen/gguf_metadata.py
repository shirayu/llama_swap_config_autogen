"""GGUF metadata reader with file-based cache."""

import json
import logging
from pathlib import Path

from gguf import GGUFReader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CACHE_PATH = Path.home() / ".cache" / "llama_swap_config_autogen" / "gguf_metadata.json"


class GGUFMetadata(BaseModel):
    mtime: float
    size: int
    num_layers: int
    num_heads: int
    num_heads_kv: int
    head_dim: int
    context_length: int
    embedding_length: int


class GGUFMetadataCache(BaseModel):
    entries: dict[str, GGUFMetadata] = Field(default_factory=dict)

    @classmethod
    def load(cls) -> "GGUFMetadataCache":
        if CACHE_PATH.exists():
            try:
                data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                cache = cls.model_validate(data)
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

    def get_int(key: str, default: int = 0) -> int:
        field = kv.get(key)
        if field is None:
            return default
        return int(field.parts[field.data[0]])

    archs = ["llama", "mistral", "phi3", "gemma", "qwen2"]
    num_layers = next((get_int(f"{a}.block_count") for a in archs if get_int(f"{a}.block_count")), 0)
    num_heads = next((get_int(f"{a}.attention.head_count") for a in archs if get_int(f"{a}.attention.head_count")), 0)
    num_heads_kv = next(
        (get_int(f"{a}.attention.head_count_kv") for a in archs if get_int(f"{a}.attention.head_count_kv")),
        num_heads,
    )
    embedding_length = next((get_int(f"{a}.embedding_length") for a in archs if get_int(f"{a}.embedding_length")), 0)
    context_length = next((get_int(f"{a}.context_length") for a in archs if get_int(f"{a}.context_length")), 0)

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
    )

    logger.debug(
        "  layers=%d, heads=%d, heads_kv=%d, head_dim=%d, ctx=%d, emb=%d",
        num_layers,
        num_heads,
        num_heads_kv,
        head_dim,
        context_length,
        embedding_length,
    )
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
    cache_type_bytes: int = 2,  # fp16 by default
) -> float:
    """Estimate VRAM usage in GB.

    Args:
        metadata: GGUF metadata
        file_size_bytes: actual file size in bytes
        ngl: number of GPU layers (-ngl value)
        context_length: context length (-c value)
        cache_type_bytes: bytes per element for KV cache (2=fp16, 1=q8, 0.5=q4)
    """
    num_layers = metadata.num_layers
    if num_layers == 0:
        logger.warning("num_layers=0, falling back to file size only")
        return file_size_bytes / 1024**3

    # Clamp ngl to [0, num_layers]
    effective_ngl = max(0, min(ngl, num_layers))
    gpu_ratio = effective_ngl / num_layers

    # Model weights on GPU
    model_vram = file_size_bytes * gpu_ratio

    # KV cache on GPU (all layers when using GPU)
    # KV = 2 (K+V) * layers * num_heads_kv * head_dim * context * bytes_per_element
    kv_vram = 0.0
    if effective_ngl > 0 and metadata.num_heads_kv > 0 and metadata.head_dim > 0:
        kv_vram = 2 * effective_ngl * metadata.num_heads_kv * metadata.head_dim * context_length * cache_type_bytes

    total_bytes = model_vram + kv_vram
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
