"""Tests for GGUF metadata cache and VRAM estimation."""

from pathlib import Path
from unittest.mock import patch

import yaml

from llama_swap_config_autogen.config import create_settings_from_config, load_config
from llama_swap_config_autogen.generator import (
    build_vram_label,
    extract_context_length,
    extract_ngl,
    generate_full_config,
)
from llama_swap_config_autogen.gguf_metadata import (
    GGUFMetadata,
    GGUFMetadataCache,
    estimate_vram_gb,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    mtime: float = 0.0,
    size: int = 4 * 1024**3,
    num_layers: int = 32,
    num_heads: int = 32,
    num_heads_kv: int = 8,
    head_dim: int = 128,
    context_length: int = 4096,
    embedding_length: int = 4096,
) -> GGUFMetadata:
    return GGUFMetadata(
        mtime=mtime,
        size=size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        context_length=context_length,
        embedding_length=embedding_length,
    )


# ---------------------------------------------------------------------------
# extract_ngl / extract_context_length
# ---------------------------------------------------------------------------


class TestExtractNgl:
    def test_found(self):
        assert extract_ngl("llama-server -m model.gguf -ngl 99 --port 8080") == 99

    def test_not_found(self):
        assert extract_ngl("llama-server -m model.gguf --port 8080") == 0

    def test_zero(self):
        assert extract_ngl("-ngl 0") == 0


class TestExtractContextLength:
    def test_short_flag(self):
        assert extract_context_length("-c 8192 --port 8080", fallback=4096) == 8192

    def test_long_flag(self):
        assert extract_context_length("--ctx-size 16384", fallback=4096) == 16384

    def test_fallback(self):
        assert extract_context_length("no context flag here", fallback=4096) == 4096


# ---------------------------------------------------------------------------
# estimate_vram_gb
# ---------------------------------------------------------------------------


class TestEstimateVramGb:
    def test_full_gpu_offload(self):
        meta = _make_metadata(num_layers=32, num_heads_kv=8, head_dim=128)
        file_size = 4 * 1024**3  # 4 GB
        result = estimate_vram_gb(meta, file_size, ngl=32, context_length=4096)
        # model: 4 GB, kv: 2*32*8*128*4096*2 bytes
        kv = 2 * 32 * 8 * 128 * 4096 * 2 / 1024**3
        assert abs(result - (4.0 + kv)) < 0.01

    def test_no_gpu_offload(self):
        meta = _make_metadata(num_layers=32)
        result = estimate_vram_gb(meta, 4 * 1024**3, ngl=0, context_length=4096)
        assert result == 0.0

    def test_partial_offload(self):
        meta = _make_metadata(num_layers=32, num_heads_kv=8, head_dim=128)
        full = estimate_vram_gb(meta, 4 * 1024**3, ngl=32, context_length=4096)
        half = estimate_vram_gb(meta, 4 * 1024**3, ngl=16, context_length=4096)
        assert half < full

    def test_ngl_clamped_to_num_layers(self):
        meta = _make_metadata(num_layers=32)
        result_exact = estimate_vram_gb(meta, 4 * 1024**3, ngl=32, context_length=4096)
        result_over = estimate_vram_gb(meta, 4 * 1024**3, ngl=999, context_length=4096)
        assert result_exact == result_over

    def test_zero_layers_fallback(self):
        meta = _make_metadata(num_layers=0)
        file_size = 4 * 1024**3
        result = estimate_vram_gb(meta, file_size, ngl=99, context_length=4096)
        assert abs(result - 4.0) < 0.001

    def test_larger_context_increases_vram(self):
        meta = _make_metadata(num_layers=32, num_heads_kv=8, head_dim=128)
        small = estimate_vram_gb(meta, 4 * 1024**3, ngl=32, context_length=4096)
        large = estimate_vram_gb(meta, 4 * 1024**3, ngl=32, context_length=32768)
        assert large > small


# ---------------------------------------------------------------------------
# GGUFMetadataCache
# ---------------------------------------------------------------------------


class TestGGUFMetadataCache:
    def test_set_and_get_hit(self, tmp_path):
        cache = GGUFMetadataCache()
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        stat = model.stat()
        meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size)
        cache.set(model, meta)
        assert cache.get(model) is not None

    def test_get_miss_file_changed(self, tmp_path):
        cache = GGUFMetadataCache()
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        stat = model.stat()
        # Store with old mtime
        meta = _make_metadata(mtime=stat.st_mtime - 1.0, size=stat.st_size)
        cache.set(model, meta)
        assert cache.get(model) is None

    def test_get_miss_size_changed(self, tmp_path):
        cache = GGUFMetadataCache()
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        stat = model.stat()
        meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size + 1)
        cache.set(model, meta)
        assert cache.get(model) is None

    def test_get_returns_none_for_unknown(self, tmp_path):
        cache = GGUFMetadataCache()
        model = tmp_path / "nonexistent.gguf"
        model.write_bytes(b"")
        assert cache.get(model) is None

    def test_save_and_load_roundtrip(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        stat = model.stat()
        meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size, num_layers=42)

        cache = GGUFMetadataCache()
        cache.set(model, meta)

        with patch("llama_swap_config_autogen.gguf_metadata.CACHE_PATH", cache_file):
            cache.save()
            loaded = GGUFMetadataCache.load()

        assert str(model) in loaded.entries
        assert loaded.entries[str(model)].num_layers == 42

    def test_load_ignores_corrupt_cache(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not json", encoding="utf-8")
        with patch("llama_swap_config_autogen.gguf_metadata.CACHE_PATH", cache_file):
            cache = GGUFMetadataCache.load()
        assert cache.entries == {}

    def test_load_returns_empty_when_no_file(self, tmp_path):
        cache_file = tmp_path / "missing.json"
        with patch("llama_swap_config_autogen.gguf_metadata.CACHE_PATH", cache_file):
            cache = GGUFMetadataCache.load()
        assert cache.entries == {}


# ---------------------------------------------------------------------------
# build_vram_label
# ---------------------------------------------------------------------------


class TestBuildVramLabel:
    def _fake_metadata(self, path: Path, cache: GGUFMetadataCache) -> GGUFMetadata:
        stat = path.stat()
        return _make_metadata(mtime=stat.st_mtime, size=stat.st_size)

    def test_returns_label_on_success(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = GGUFMetadataCache()

        with patch(
            "llama_swap_config_autogen.generator.get_gguf_metadata",
            side_effect=self._fake_metadata,
        ):
            label = build_vram_label(model, "-ngl 32 -c 4096", 4096, cache)

        assert label is not None
        assert label.startswith("[")
        assert label.endswith(" GB]")

    def test_returns_none_on_error(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = GGUFMetadataCache()

        with patch(
            "llama_swap_config_autogen.generator.get_gguf_metadata",
            side_effect=RuntimeError("read error"),
        ):
            label = build_vram_label(model, "-ngl 32 -c 4096", 4096, cache)

        assert label is None

    def test_label_format(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * (4 * 1024**3 // 100))  # small file for speed
        cache = GGUFMetadataCache()
        stat = model.stat()
        meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size)

        with patch("llama_swap_config_autogen.generator.get_gguf_metadata", return_value=meta):
            label = build_vram_label(model, "-ngl 32 -c 4096", 4096, cache)

        import re

        assert label is not None
        assert re.match(r"^\[\d+\.\d GB\]$", label)


# ---------------------------------------------------------------------------
# Integration: model name contains VRAM label
# ---------------------------------------------------------------------------


def _write_config(config_path: Path, models_dir: Path, vram_estimation: bool = True, extra: dict | None = None) -> None:
    config: dict = {
        "models": [str(models_dir)],
        "vram_estimation": vram_estimation,
        "macros": {
            "binary": "/app/llama-server",
            "default-params": "-ngl 99 --ctx-size 4096",
        },
    }
    if extra:
        config.update(extra)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class TestVramLabelInGeneratedConfig:
    def _fake_get_metadata(self, path: Path, cache: GGUFMetadataCache) -> GGUFMetadata:
        stat = path.stat()
        return _make_metadata(mtime=stat.st_mtime, size=stat.st_size)

    def test_model_name_includes_vram_label(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, models_dir)

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)

        with patch("llama_swap_config_autogen.generator.get_gguf_metadata", side_effect=self._fake_get_metadata):
            result = generate_full_config(settings, config)

        names = [v["name"] for v in result["models"].values()]
        assert any("[" in name and "GB]" in name for name in names), f"No VRAM label found in names: {names}"

    def test_model_name_no_label_when_disabled(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, models_dir, vram_estimation=False)

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)

        result = generate_full_config(settings, config)

        names = [v["name"] for v in result["models"].values()]
        assert all("[" not in name for name in names), f"Unexpected VRAM label in names: {names}"

    def test_model_name_no_label_on_read_error(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, models_dir)

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)

        with patch(
            "llama_swap_config_autogen.generator.get_gguf_metadata",
            side_effect=RuntimeError("gguf read failed"),
        ):
            result = generate_full_config(settings, config)

        names = [v["name"] for v in result["models"].values()]
        assert all("[" not in name for name in names), f"Unexpected VRAM label in names: {names}"

    def test_cache_is_saved_after_new_entry(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "llama3"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "llama3-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(config_path, models_dir)

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)

        saved = {}

        def fake_get(path, cache):
            stat = path.stat()
            meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size)
            cache.set(path, meta)  # simulate a cache miss (new entry)
            return meta

        def fake_save(self_cache):
            saved["called"] = True

        with (
            patch("llama_swap_config_autogen.generator.get_gguf_metadata", side_effect=fake_get),
            patch.object(GGUFMetadataCache, "save", fake_save),
        ):
            generate_full_config(settings, config)

        assert saved.get("called"), "cache.save() was not called after new entries were added"
