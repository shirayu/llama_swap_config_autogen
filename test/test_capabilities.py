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
    assert output["models"][model_id]["capabilities"] == {"context": 32768}


def test_no_capabilities_when_context_cannot_be_determined(tmp_path: Path):
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
    assert "capabilities" not in output["models"][model_id]


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
    assert output["models"][base_id]["capabilities"] == {"context": 32768}
    assert output["models"][variant_id]["capabilities"] == {"context": 65536}


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
    assert output["models"][model_id]["capabilities"] == {"context": 8192}
