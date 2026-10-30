"""Tests for configuration loading diagnostics."""

import pytest

from llama_swap_config_autogen.config import create_settings_from_config, load_config, load_macro_config


def test_invalid_model_pattern_reports_yaml_location(tmp_path):
    config_file = tmp_path / "base.yaml"
    config_file.write_text(
        "models:\n  - /models\nmodel_patterns:\n  broken:\n    emit_base: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"{config_file}:4:.*broken.*macro"):
        load_macro_config(config_file)


def test_invalid_config_reports_field_and_yaml_location(tmp_path):
    config_file = tmp_path / "base.yaml"
    config_file.write_text("models: not-a-list\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{config_file}:1 \(models\):.*list"):
        load_config(config_file)


class TestVramCorrectionDefaults:
    def test_correction_is_on_by_default(self, tmp_path):
        config_file = tmp_path / "base.yaml"
        config_file.write_text(f"models:\n  - {tmp_path}\n", encoding="utf-8")
        settings = create_settings_from_config(load_config(config_file), config_file)
        assert settings.vram_overhead_mib == 500
        assert settings.mmproj_vram_extra_mib == 150
        assert settings.vram_default_parallel == 4

    def test_correction_can_be_disabled(self, tmp_path):
        config_file = tmp_path / "base.yaml"
        config_file.write_text(
            f"models:\n  - {tmp_path}\nvram_overhead_mib: 0\nmmproj_vram_extra_mib: 0\nvram_default_parallel: null\n",
            encoding="utf-8",
        )
        settings = create_settings_from_config(load_config(config_file), config_file)
        assert settings.vram_overhead_mib == 0
        assert settings.mmproj_vram_extra_mib == 0
        assert settings.vram_default_parallel is None
