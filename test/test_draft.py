"""Tests for draft (speculative decoding / MTP) model handling in config generation."""

from pathlib import Path

import yaml

from llama_swap_config_autogen.config import create_settings_from_config, load_config
from llama_swap_config_autogen.generator import generate_full_config


def _touch(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _write_base_config(
    config_path: Path,
    models_dir: Path,
    draft: dict | None = None,
    extra: dict | None = None,
) -> None:
    config = {
        "models": [str(models_dir)],
        "macros": {
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
    }
    if draft is not None:
        config["draft"] = draft
    if extra is not None:
        config.update(extra)

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_draft_is_not_auto_attached_by_default(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.8-27B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "Qwen3.8-27B-Q4_K_M.gguf"
    draft_file = target_dir / "mtp-Qwen3.8-27B-Q4_0.gguf"
    _touch(model_file)
    _touch(draft_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3.8-27b/standard:Q4_K_M"
    assert model_id in output["models"]
    assert "--model-draft" not in output["models"][model_id]["cmd"]
    # The draft file itself must never be emitted as a standalone model.
    assert not any("mtp" in key.lower() for key in output["models"].keys())


def test_draft_auto_attach_when_enabled(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.8-27B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "Qwen3.8-27B-Q4_K_M.gguf"
    draft_file = target_dir / "mtp-Qwen3.8-27B-Q4_0.gguf"
    _touch(model_file)
    _touch(draft_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir, draft={"enabled": True, "auto_attach": True})

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3.8-27b/standard:Q4_K_M"
    assert model_id in output["models"]
    assert "--model-draft" in output["models"][model_id]["cmd"]
    assert str(draft_file) in output["models"][model_id]["cmd"]
    assert not any("mtp" in key.lower() for key in output["models"].keys())


def test_draft_override_applies_when_auto_attach_is_disabled(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.8-27B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "Qwen3.8-27B-Q4_K_M.gguf"
    override_draft = target_dir / "mtp-Qwen3.8-27B-Q4_0.gguf"
    _touch(model_file)
    _touch(override_draft)

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        draft={
            "enabled": True,
            "auto_attach": False,
            "overrides": {
                "qwen3.8-27b/standard:Q4_K_M": str(override_draft),
            },
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3.8-27b/standard:Q4_K_M"
    assert model_id in output["models"]
    assert "--model-draft" in output["models"][model_id]["cmd"]
    assert str(override_draft) in output["models"][model_id]["cmd"]


def test_draft_via_model_pattern(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.8-27B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "Qwen3.8-27B-Q4_K_M.gguf"
    draft_file = target_dir / "mtp-Qwen3.8-27B-Q4_0.gguf"
    _touch(model_file)
    _touch(draft_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        extra={
            "model_patterns": {
                "qwen3.8-27b": {
                    "macro": "default-params",
                    "draft": "mtp-Qwen3.8-27B-Q4_0.gguf",
                }
            }
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3.8-27b/standard:Q4_K_M"
    assert model_id in output["models"]
    assert "--model-draft" in output["models"][model_id]["cmd"]
    assert str(draft_file) in output["models"][model_id]["cmd"]


def test_draft_not_auto_attached_when_multiple_candidates_exist(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.8-27B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "Qwen3.8-27B-Q4_K_M.gguf"
    draft_file_a = target_dir / "mtp-Qwen3.8-27B-Q4_0.gguf"
    draft_file_b = target_dir / "mtp-Qwen3.8-27B-Q8_0.gguf"
    _touch(model_file)
    _touch(draft_file_a)
    _touch(draft_file_b)

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir, draft={"enabled": True, "auto_attach": True})

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3.8-27b/standard:Q4_K_M"
    assert model_id in output["models"]
    assert "--model-draft" not in output["models"][model_id]["cmd"]
