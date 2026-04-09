"""Tests for GGUF metadata cache and VRAM estimation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from llama_swap_config_autogen.config import create_settings_from_config, load_config
from llama_swap_config_autogen.generator import (
    build_vram_label,
    estimate_cpu_offload_gpu_ratio,
    expand_macro_expression,
    extract_cache_type_bytes,
    extract_context_length,
    extract_ngl,
    generate_full_config,
)
from llama_swap_config_autogen.gguf_metadata import (
    GGUFMetadata,
    GGUFMetadataCache,
    _read_gguf_metadata,
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
    expert_count: int = 0,
    expert_used_count: int = 0,
    feed_forward_length: int = 0,
    expert_feed_forward_length: int = 0,
    expert_shared_feed_forward_length: int = 0,
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
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        feed_forward_length=feed_forward_length,
        expert_feed_forward_length=expert_feed_forward_length,
        expert_shared_feed_forward_length=expert_shared_feed_forward_length,
    )


# ---------------------------------------------------------------------------
# extract_ngl / extract_context_length
# ---------------------------------------------------------------------------


class TestExtractNgl:
    def test_found(self):
        assert extract_ngl("llama-server -m model.gguf -ngl 99 --port 8080") == 99

    def test_found_long_flag(self):
        assert extract_ngl("llama-server -m model.gguf --n-gpu-layers 25 --port 8080") == 25

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


class TestExtractCacheTypeBytes:
    def test_defaults_to_fp16(self):
        assert extract_cache_type_bytes("--ctx-size 4096") == (2.0, 2.0)

    def test_extracts_q8_and_q4(self):
        assert extract_cache_type_bytes("--cache-type-k q8_0 --cache-type-v q4_0") == (1.0, 0.5)


class TestCpuOffloadEstimation:
    def test_returns_none_without_moe_metadata(self):
        meta = _make_metadata()
        assert estimate_cpu_offload_gpu_ratio(meta, "--n-cpu-moe 12") is None

    def test_estimates_partial_cpu_offload(self):
        meta = _make_metadata(
            expert_count=128,
            expert_used_count=8,
            feed_forward_length=6144,
            expert_feed_forward_length=768,
        )
        ratio = estimate_cpu_offload_gpu_ratio(meta, "--n-cpu-moe 12")
        assert ratio is not None
        assert 0.8 < ratio < 1.0


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

    def test_quantized_kv_cache_reduces_vram(self):
        meta = _make_metadata(num_layers=32, num_heads_kv=8, head_dim=128)
        fp16 = estimate_vram_gb(meta, 4 * 1024**3, ngl=32, context_length=32768)
        quantized = estimate_vram_gb(
            meta,
            4 * 1024**3,
            ngl=32,
            context_length=32768,
            k_cache_bytes=1.0,
            v_cache_bytes=0.5,
        )
        assert quantized < fp16

    def test_mmproj_increases_vram(self):
        meta = _make_metadata(num_layers=32, num_heads_kv=8, head_dim=128)
        base = estimate_vram_gb(meta, 4 * 1024**3, ngl=32, context_length=4096)
        with_mmproj = estimate_vram_gb(meta, 4 * 1024**3, ngl=32, context_length=4096, extra_gpu_bytes=512 * 1024**2)
        assert with_mmproj > base


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


class TestReadGgufMetadata:
    def test_handles_array_like_field_contents(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00")

        class FakeField:
            def __init__(self, name, value):
                self.name = name
                self._value = value

            def contents(self, index_or_slice=0):
                return self._value

        fake_fields = {
            "qwen2.block_count": FakeField("qwen2.block_count", [32]),
            "qwen2.attention.head_count": FakeField("qwen2.attention.head_count", [32]),
            "qwen2.attention.head_count_kv": FakeField("qwen2.attention.head_count_kv", [8]),
            "qwen2.embedding_length": FakeField("qwen2.embedding_length", [4096]),
            "qwen2.context_length": FakeField("qwen2.context_length", [131072]),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            meta = _read_gguf_metadata(model)

        assert meta.num_layers == 32
        assert meta.num_heads == 32
        assert meta.num_heads_kv == 8
        assert meta.embedding_length == 4096
        assert meta.context_length == 131072
        assert meta.head_dim == 128

    def test_discovers_non_fallback_arch_prefix(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00")

        class FakeField:
            def __init__(self, name, value):
                self.name = name
                self._value = value

            def contents(self, index_or_slice=0):
                return self._value

        fake_fields = {
            "qwen3.block_count": FakeField("qwen3.block_count", 48),
            "qwen3.attention.head_count": FakeField("qwen3.attention.head_count", 40),
            "qwen3.attention.head_count_kv": FakeField("qwen3.attention.head_count_kv", 8),
            "qwen3.embedding_length": FakeField("qwen3.embedding_length", 5120),
            "qwen3.context_length": FakeField("qwen3.context_length", 40960),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            meta = _read_gguf_metadata(model)

        assert meta.num_layers == 48
        assert meta.num_heads == 40
        assert meta.num_heads_kv == 8
        assert meta.embedding_length == 5120
        assert meta.context_length == 40960
        assert meta.head_dim == 128

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
        assert re.match(r"^\[\d+\.\d GB\??\]$", label)

    def test_low_confidence_label_has_question_mark(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 100)
        cache = GGUFMetadataCache()
        stat = model.stat()
        meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size, head_dim=0)

        with patch("llama_swap_config_autogen.generator.get_gguf_metadata", return_value=meta):
            label = build_vram_label(model, "--n-gpu-layers 32 -c 4096", 4096, cache)

        assert label is not None
        assert label.endswith("GB?]")

    def test_mmproj_is_reflected_in_label(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * 1024)
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"\x00" * (512 * 1024**2))
        cache = GGUFMetadataCache()
        stat = model.stat()
        meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size)

        with patch("llama_swap_config_autogen.generator.get_gguf_metadata", return_value=meta):
            base = build_vram_label(model, "--n-gpu-layers 32 -c 4096", 4096, cache)
            with_mmproj = build_vram_label(model, "--n-gpu-layers 32 -c 4096", 4096, cache, mmproj_path=mmproj)

        assert base is not None and with_mmproj is not None
        assert with_mmproj != base

    def test_cpu_offload_label_can_be_high_confidence(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00" * (4 * 1024**3 // 100))
        cache = GGUFMetadataCache()
        stat = model.stat()
        meta = _make_metadata(
            mtime=stat.st_mtime,
            size=stat.st_size,
            expert_count=128,
            expert_used_count=8,
            feed_forward_length=6144,
            expert_feed_forward_length=768,
        )

        with patch("llama_swap_config_autogen.generator.get_gguf_metadata", return_value=meta):
            base = build_vram_label(model, "--n-gpu-layers 32 --ctx-size 65536", 4096, cache)
            cpu = build_vram_label(model, "--n-gpu-layers 32 --n-cpu-moe 12 --ctx-size 65536", 4096, cache)

        assert base is not None and cpu is not None
        assert "?" not in cpu
        ratio = estimate_cpu_offload_gpu_ratio(meta, "--n-cpu-moe 12")
        assert ratio is not None
        assert ratio < 1.0


class TestExpandMacroExpression:
    def test_expands_plain_macro_name(self):
        macros = {"default-params": "--n-gpu-layers 999 --ctx-size 32768"}
        assert expand_macro_expression("default-params", macros) == "--n-gpu-layers 999 --ctx-size 32768"

    def test_expands_composite_expression(self):
        macros = {
            "deepseek-r1-params": "--n-gpu-layers 999 --ctx-size 32768",
            "inference-fast": "--batch-size 512 --ubatch-size 512",
        }
        result = expand_macro_expression("${deepseek-r1-params} ${inference-fast}", macros)
        assert result == "--n-gpu-layers 999 --ctx-size 32768 --batch-size 512 --ubatch-size 512"


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

    def test_variant_composite_macro_includes_vram_label(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "deepseek-r1-distill-qwen-32b"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "distill-Q4_K_M.gguf"
        model_file.write_bytes(b"\x00" * 1024)

        config_path = tmp_path / "config.yaml"
        _write_config(
            config_path,
            models_dir,
            extra={
                "model_patterns": {"deepseek-r1-distill-qwen-32b": "default-params"},
                "variants": [
                    {
                        "base_pattern": "r1-distill",
                        "suffix": " (Fast)",
                        "macro": "${default-params} --batch-size 512",
                    }
                ],
            },
        )

        config = load_config(config_path)
        settings = create_settings_from_config(config, config_path)

        with patch("llama_swap_config_autogen.generator.get_gguf_metadata", side_effect=self._fake_get_metadata):
            result = generate_full_config(settings, config)

        fast_names = [v["name"] for model_id, v in result["models"].items() if model_id.endswith("-fast")]
        assert fast_names, "Fast variant was not generated"
        assert all("GB]" in name for name in fast_names), f"Variant VRAM label missing: {fast_names}"
