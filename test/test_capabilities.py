"""Tests for automatic capabilities.context generation."""

from pathlib import Path

import yaml

from llama_swap_config_autogen.config import create_settings_from_config, load_config
from llama_swap_config_autogen.generator import generate_full_config


def _touch(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _write_base_config(config_path: Path, models_dir: Path, macros: dict, extra: dict | None = None) -> None:
    config = {
        "models": [str(models_dir)],
        "macros": macros,
    }
    if extra is not None:
        config.update(extra)

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_context_length_from_ctx_size_flag_is_added_to_capabilities(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"] == {"context": 32768, "in": ["text"]}


def test_only_in_text_when_context_cannot_be_determined(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja",
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"] == {"in": ["text"]}


def test_variant_uses_its_own_context_length(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
            "large-ctx": "--jinja --ctx-size 65536",
        },
        extra={
            "variants": [
                {
                    "base_pattern": "qwen3-30b",
                    "suffix": " (64k)",
                    "macro": "large-ctx",
                }
            ]
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    base_id = "qwen3-30b/standard:Q4_K_M"
    variant_id = "qwen3-30b/standard:Q4_K_M--64k"
    assert output["models"][base_id]["capabilities"] == {"context": 32768, "in": ["text"]}
    assert output["models"][variant_id]["capabilities"] == {"context": 65536, "in": ["text"]}


def test_user_specified_capabilities_merge_with_auto_derived_context(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
        extra={
            "model_patterns": {
                "qwen3-30b": {
                    "macro": "default-params",
                    "capabilities": {
                        "in": ["text", "image"],
                        "out": ["text"],
                        "tools": True,
                        "reranker": False,
                    },
                }
            }
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"] == {
        "in": ["text", "image"],
        "out": ["text"],
        "tools": True,
        "reranker": False,
        "context": 32768,
    }


def test_user_specified_context_takes_priority_over_auto_derived_context(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
        extra={
            "model_patterns": {
                "qwen3-30b": {
                    "macro": "default-params",
                    "capabilities": {"context": 8192},
                }
            }
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"] == {"context": 8192, "in": ["text"]}


def test_in_includes_image_when_mmproj_is_attached(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")
    _touch(target_dir / "mmproj-F16.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"]["in"] == ["text", "image"]


def test_user_specified_in_overrides_auto_derived_in(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")
    _touch(target_dir / "mmproj-F16.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
        extra={
            "model_patterns": {
                "qwen3-30b": {
                    "macro": "default-params",
                    "capabilities": {"in": ["text"]},
                }
            }
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"]["in"] == ["text"]


def test_tools_detected_from_gguf_chat_template(tmp_path: Path, monkeypatch):
    from llama_swap_config_autogen.gguf_metadata import GGUFMetadata

    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
        extra={"vram_estimation": True},
    )

    def fake_get_gguf_metadata(path, cache):
        return GGUFMetadata(
            mtime=0,
            size=0,
            num_layers=1,
            num_heads=1,
            num_heads_kv=1,
            head_dim=1,
            context_length=32768,
            embedding_length=1,
            supports_tools=True,
        )

    monkeypatch.setattr("llama_swap_config_autogen.generator.get_gguf_metadata", fake_get_gguf_metadata)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"]["tools"] is True


def test_user_specified_tools_overrides_auto_derived_tools(tmp_path: Path, monkeypatch):
    from llama_swap_config_autogen.gguf_metadata import GGUFMetadata

    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3-30B" / "standard"
    target_dir.mkdir(parents=True)
    _touch(target_dir / "model-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        macros={
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
        extra={
            "vram_estimation": True,
            "model_patterns": {
                "qwen3-30b": {
                    "macro": "default-params",
                    "capabilities": {"tools": False},
                }
            },
        },
    )

    def fake_get_gguf_metadata(path, cache):
        return GGUFMetadata(
            mtime=0,
            size=0,
            num_layers=1,
            num_heads=1,
            num_heads_kv=1,
            head_dim=1,
            context_length=32768,
            embedding_length=1,
            supports_tools=True,
        )

    monkeypatch.setattr("llama_swap_config_autogen.generator.get_gguf_metadata", fake_get_gguf_metadata)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3-30b/standard:Q4_K_M"
    assert output["models"][model_id]["capabilities"]["tools"] is False
