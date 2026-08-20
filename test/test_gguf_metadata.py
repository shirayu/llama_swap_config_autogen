"""Tests for GGUF metadata cache."""

from types import SimpleNamespace
from unittest.mock import patch

from llama_swap_config_autogen.gguf_metadata import (
    GGUFMetadata,
    GGUFMetadataCache,
    _read_gguf_metadata,
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
    full_attention_interval: int = 0,
    supports_reasoning: bool = False,
    repo_url: str = "",
    license: str = "",
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
        full_attention_interval=full_attention_interval,
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

        assert meta.num_layers == 48
        assert meta.num_heads == 40
        assert meta.num_heads_kv == 8
        assert meta.embedding_length == 5120
        assert meta.context_length == 40960

    def test_prefers_explicit_key_length_over_derived_head_dim(self, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"\x00")

        class FakeField:
            def __init__(self, name, value):
                self.name = name
                self._value = value

            def contents(self, index_or_slice=0):
                return self._value

        # embedding_length // head_count would give 213, but the GGUF explicitly
        # states a head_dim of 256 (as seen on hybrid SSM/attention Qwen3.5 models).
        fake_fields = {
            "qwen35.block_count": FakeField("qwen35.block_count", [65]),
            "qwen35.attention.head_count": FakeField("qwen35.attention.head_count", [24]),
            "qwen35.attention.head_count_kv": FakeField("qwen35.attention.head_count_kv", [4]),
            "qwen35.attention.key_length": FakeField("qwen35.attention.key_length", [256]),
            "qwen35.embedding_length": FakeField("qwen35.embedding_length", [5120]),
            "qwen35.context_length": FakeField("qwen35.context_length", [262144]),
            "qwen35.full_attention_interval": FakeField("qwen35.full_attention_interval", [4]),
        }
        fake_reader = SimpleNamespace(fields=fake_fields)

        with patch("llama_swap_config_autogen.gguf_metadata.GGUFReader", return_value=fake_reader):
            meta = _read_gguf_metadata(model)

        assert meta.head_dim == 256
        assert meta.full_attention_interval == 4

    def test_falls_back_to_derived_head_dim_without_key_length(self, tmp_path):
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

        assert meta.head_dim == 128
        assert meta.full_attention_interval == 0
        assert meta.head_dim == 128

    def test_load_returns_empty_when_no_file(self, tmp_path):
        cache_file = tmp_path / "missing.json"
        with patch("llama_swap_config_autogen.gguf_metadata.CACHE_PATH", cache_file):
            cache = GGUFMetadataCache.load()
        assert cache.entries == {}
