"""Tests for GGUF metadata cache."""

from types import SimpleNamespace
from unittest.mock import patch

from llama_swap_config_autogen.gguf_metadata import (
    GGUFMetadata,
    GGUFMetadataCache,
    _read_gguf_metadata,
    read_mmproj_modalities,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    mtime: float = 0.0,
    size: int = 4 * 1024**3,
    context_length: int = 4096,
    expert_count: int = 0,
    expert_used_count: int = 0,
    supports_reasoning: bool = False,
    repo_url: str = "",
    license: str = "",
) -> GGUFMetadata:
    return GGUFMetadata(
        mtime=mtime,
        size=size,
        context_length=context_length,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        supports_reasoning=supports_reasoning,
        repo_url=repo_url,
        license=license,
    )


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
        meta = _make_metadata(mtime=stat.st_mtime, size=stat.st_size, context_length=131072)

        cache = GGUFMetadataCache()
        cache.set(model, meta)

        with patch("llama_swap_config_autogen.gguf_metadata.CACHE_PATH", cache_file):
            cache.save()
            loaded = GGUFMetadataCache.load()

        assert str(model) in loaded.entries
        assert loaded.entries[str(model)].context_length == 131072

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

        assert meta.context_length == 131072

    def test_detects_reasoning_support_from_chat_template(self, tmp_path):
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
            "tokenizer.chat_template": FakeField(
                "tokenizer.chat_template",
                "{% if enable_thinking is defined and enable_thinking is false %}{% endif %}",
            ),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            meta = _read_gguf_metadata(model)

        assert meta.supports_reasoning is True

    def test_reasoning_support_false_without_marker(self, tmp_path):
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
            "tokenizer.chat_template": FakeField("tokenizer.chat_template", "{{ messages }}"),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            meta = _read_gguf_metadata(model)

        assert meta.supports_reasoning is False

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

        assert meta.context_length == 40960

    def test_load_returns_empty_when_no_file(self, tmp_path):
        cache_file = tmp_path / "missing.json"
        with patch("llama_swap_config_autogen.gguf_metadata.CACHE_PATH", cache_file):
            cache = GGUFMetadataCache.load()
        assert cache.entries == {}


class TestReadMmprojModalities:
    def test_detects_vision_encoder(self, tmp_path):
        mmproj = tmp_path / "mmproj-F16.gguf"
        mmproj.write_bytes(b"\x00")

        class FakeField:
            def __init__(self, name, value):
                self.name = name
                self._value = value

            def contents(self, index_or_slice=0):
                return self._value

        fake_fields = {
            "clip.has_vision_encoder": FakeField("clip.has_vision_encoder", [True]),
            "clip.has_audio_encoder": FakeField("clip.has_audio_encoder", [False]),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            modalities = read_mmproj_modalities(mmproj)

        assert modalities.has_vision is True
        assert modalities.has_audio is False

    def test_detects_audio_encoder(self, tmp_path):
        mmproj = tmp_path / "mmproj-ultravox-F16.gguf"
        mmproj.write_bytes(b"\x00")

        class FakeField:
            def __init__(self, name, value):
                self.name = name
                self._value = value

            def contents(self, index_or_slice=0):
                return self._value

        fake_fields = {
            "clip.has_vision_encoder": FakeField("clip.has_vision_encoder", [False]),
            "clip.has_audio_encoder": FakeField("clip.has_audio_encoder", [True]),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            modalities = read_mmproj_modalities(mmproj)

        assert modalities.has_vision is False
        assert modalities.has_audio is True

    def test_defaults_to_false_when_keys_missing(self, tmp_path):
        mmproj = tmp_path / "mmproj-F16.gguf"
        mmproj.write_bytes(b"\x00")

        fake_reader = SimpleNamespace(fields={})

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            modalities = read_mmproj_modalities(mmproj)

        assert modalities.has_vision is False
        assert modalities.has_audio is False

    def test_returns_defaults_when_file_cannot_be_opened(self, tmp_path):
        missing = tmp_path / "does-not-exist.gguf"
        modalities = read_mmproj_modalities(missing)
        assert modalities.has_vision is False
        assert modalities.has_audio is False

    def test_reads_projector_type(self, tmp_path):
        mmproj = tmp_path / "mmproj-ultravox-F16.gguf"
        mmproj.write_bytes(b"\x00")

        class FakeField:
            def __init__(self, name, value):
                self.name = name
                self._value = value

            def contents(self, index_or_slice=0):
                return self._value

        fake_fields = {
            "clip.has_audio_encoder": FakeField("clip.has_audio_encoder", [True]),
            "clip.projector_type": FakeField("clip.projector_type", "ultravox"),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            modalities = read_mmproj_modalities(mmproj)

        assert modalities.projector_type == "ultravox"

    def test_projector_type_defaults_to_empty_string(self, tmp_path):
        mmproj = tmp_path / "mmproj-F16.gguf"
        mmproj.write_bytes(b"\x00")

        fake_reader = SimpleNamespace(fields={})

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            modalities = read_mmproj_modalities(mmproj)

        assert modalities.projector_type == ""
