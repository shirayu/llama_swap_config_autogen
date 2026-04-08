"""Tests for mmproj handling in config generation."""

from pathlib import Path

import yaml

from llama_swap_config_autogen.config import create_settings_from_config, load_config
from llama_swap_config_autogen.generator import generate_full_config


def _touch(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _write_base_config(
    config_path: Path,
    models_dir: Path,
    mmproj: dict | None = None,
    extra: dict | None = None,
) -> None:
    config = {
        "models": [str(models_dir)],
        "macros": {
            "binary": "/app/llama-server",
            "default-params": "--jinja --ctx-size 32768",
        },
    }
    if mmproj is not None:
        config["mmproj"] = mmproj
    if extra is not None:
        config.update(extra)

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_mmproj_auto_attach_and_skip_as_standalone_model(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.5-35B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
    mmproj_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_mmproj-F16.gguf"
    _touch(model_file)
    _touch(mmproj_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3.5-35b/standard:Q4_K_M"
    assert model_id in output["models"]
    assert "--mmproj" in output["models"][model_id]["cmd"]
    assert str(mmproj_file) in output["models"][model_id]["cmd"]
    assert all(":F16" not in key for key in output["models"].keys())
    no_mmproj_entries = [k for k, v in output["models"].items() if v["name"].endswith("(no mmproj)")]
    assert len(no_mmproj_entries) == 1
    assert "--mmproj" not in output["models"][no_mmproj_entries[0]]["cmd"]


def test_mmproj_override_applies_when_auto_attach_is_disabled(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.5-35B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
    override_mmproj = target_dir / "custom_mmproj_file.gguf"
    _touch(model_file)
    _touch(override_mmproj)

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        mmproj={
            "enabled": True,
            "auto_attach": False,
            "overrides": {
                "qwen3.5-35b/standard:Q4_K_M": str(override_mmproj),
            },
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "qwen3.5-35b/standard:Q4_K_M"
    assert model_id in output["models"]
    assert "--mmproj" in output["models"][model_id]["cmd"]
    assert str(override_mmproj) in output["models"][model_id]["cmd"]


def test_mmproj_can_be_disabled_to_keep_legacy_behavior(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.5-35B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
    mmproj_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_mmproj-F16.gguf"
    _touch(model_file)
    _touch(mmproj_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        mmproj={
            "enabled": False,
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    assert "qwen3.5-35b/standard:Q4_K_M" in output["models"]
    assert "qwen3.5-35b/standard:F16" in output["models"]
    assert "--mmproj" not in output["models"]["qwen3.5-35b/standard:Q4_K_M"]["cmd"]


def test_generate_no_mmproj_variant_for_model_and_variants(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen3.5-35B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
    mmproj_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_mmproj-F16.gguf"
    _touch(model_file)
    _touch(mmproj_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(
        config_path,
        models_dir,
        mmproj={
            "enabled": True,
            "auto_attach": True,
            "generate_no_mmproj_variant": True,
            "no_mmproj_suffix": " (no mmproj)",
        },
        extra={
            "variants": [
                {
                    "base_pattern": "qwen3.5-35b",
                    "suffix": " (short ctx)",
                    "macro": "default-params",
                }
            ]
        },
    )

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)
    model_entries = output["models"]

    primary_id = "qwen3.5-35b/standard:Q4_K_M"
    assert primary_id in model_entries
    assert "--mmproj" in model_entries[primary_id]["cmd"]

    no_mmproj_entries = [k for k, v in model_entries.items() if v["name"].endswith("(no mmproj)")]
    assert len(no_mmproj_entries) == 2
    for key in no_mmproj_entries:
        assert "--mmproj" not in model_entries[key]["cmd"]


def test_mmproj_prefers_single_bf16_candidate_when_multiple_candidates_exist(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Gemma-4-31B" / "it"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "unsloth_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q4_K_M.gguf"
    bf16_mmproj_file = target_dir / "unsloth_gemma-4-31B-it-GGUF_mmproj-BF16.gguf"
    f16_mmproj_file = target_dir / "unsloth_gemma-4-31B-it-GGUF_mmproj-F16.gguf"
    _touch(model_file)
    _touch(bf16_mmproj_file)
    _touch(f16_mmproj_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    model_id = "gemma-4-31b/it:Q4_K_M"
    assert model_id in output["models"]
    assert "--mmproj" in output["models"][model_id]["cmd"]
    assert str(bf16_mmproj_file) in output["models"][model_id]["cmd"]
    assert str(f16_mmproj_file) not in output["models"][model_id]["cmd"]


def test_generation_fails_when_model_is_stored_at_unexpected_depth(tmp_path: Path):
    models_dir = tmp_path / "models"
    unexpected_dir = models_dir
    unexpected_dir.mkdir(parents=True)

    model_file = unexpected_dir / "Qwen_Qwen3-1.7B-GGUF_Qwen3-1.7B-Q8_0.gguf"
    _touch(model_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)

    try:
        generate_full_config(settings, config)
    except ValueError as exc:
        assert "Unexpected model directory depth" in str(exc)
    else:
        raise AssertionError("Expected generation to fail for unexpected directory depth")


def test_family_directory_is_ignored_when_all_models_use_family_layout(tmp_path: Path):
    models_dir = tmp_path / "models"
    target_dir = models_dir / "Qwen" / "Qwen3.5-35B" / "standard"
    target_dir.mkdir(parents=True)

    model_file = target_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
    _touch(model_file)

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)
    output = generate_full_config(settings, config)

    assert "qwen3.5-35b/standard:Q4_K_M" in output["models"]


def test_generation_fails_when_family_and_non_family_layouts_are_mixed(tmp_path: Path):
    models_dir = tmp_path / "models"
    family_dir = models_dir / "Qwen" / "Qwen3.5-35B" / "standard"
    direct_dir = models_dir / "Gemma-4-31B" / "it"
    family_dir.mkdir(parents=True)
    direct_dir.mkdir(parents=True)

    _touch(family_dir / "unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf")
    _touch(direct_dir / "unsloth_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q4_K_M.gguf")

    config_path = tmp_path / "base.yaml"
    _write_base_config(config_path, models_dir)

    config = load_config(config_path)
    settings = create_settings_from_config(config, config_path)

    try:
        generate_full_config(settings, config)
    except ValueError as exc:
        assert "Unexpected mixed model directory depths" in str(exc)
    else:
        raise AssertionError("Expected generation to fail for mixed family and non-family layouts")
