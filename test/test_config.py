"""Tests for configuration loading diagnostics."""

import pytest

from llama_swap_config_autogen.config import load_config, load_macro_config


def test_invalid_model_pattern_reports_yaml_location(tmp_path):
    config_file = tmp_path / "base.yaml"
    config_file.write_text(
        "models:\n"
        "  - /models\n"
        "model_patterns:\n"
        "  broken:\n"
        "    emit_base: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"{config_file}:4:.*broken.*macro"):
        load_macro_config(config_file)


def test_invalid_config_reports_field_and_yaml_location(tmp_path):
    config_file = tmp_path / "base.yaml"
    config_file.write_text("models: not-a-list\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{config_file}:1 \(models\):.*list"):
        load_config(config_file)
