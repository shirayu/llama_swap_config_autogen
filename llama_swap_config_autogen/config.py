"""Configuration loading and processing functions."""

from pathlib import Path

import yaml

from .models import Config, MacroConfig, Settings


def load_config(config_file: Path) -> Config:
    """Load configuration file"""
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open() as f:
        raw_data = yaml.safe_load(f)

    if not raw_data:
        raise ValueError(f"Empty config file: {config_file}")

    return Config(**raw_data)


def load_macro_config(config_file: Path) -> MacroConfig:
    """Load macros configuration file"""
    if not config_file.exists():
        return MacroConfig()

    with config_file.open() as f:
        raw_data = yaml.safe_load(f)

    if not raw_data:
        return MacroConfig()

    return MacroConfig(
        macros=raw_data.get("macros", {}),
        model_patterns=raw_data.get("model_patterns", {}),
        variants=raw_data.get("variants", []),
    )


def create_settings_from_config(config: Config, config_file: Path) -> Settings:
    """Create Settings object from configuration."""
    return Settings(
        models_dirs=config.models,
        default_ttl=config.default_ttl,
        health_check_timeout=config.health_check_timeout,
        log_level=config.log_level,
        start_port=config.start_port,
        config_file=config_file,
    )
