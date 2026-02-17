"""Configuration loading and processing functions."""

import re
from pathlib import Path

import yaml

from .models import Config, MacroConfig, Settings

MACRO_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


def normalize_macro_name(name: str) -> str:
    """Normalize macro name for llama-swap compatibility."""
    return name.replace(".", "-")


def normalize_macro_references(value: str, name_map: dict[str, str]) -> str:
    """Normalize macro references inside a macro expression."""

    def replace_ref(match: re.Match[str]) -> str:
        ref_name = match.group(1)
        normalized_ref = name_map.get(ref_name, normalize_macro_name(ref_name))
        return f"${{{normalized_ref}}}"

    return MACRO_REF_PATTERN.sub(replace_ref, value)


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

    raw_macros: dict[str, str] = raw_data.get("macros", {})
    name_map = {name: normalize_macro_name(name) for name in raw_macros}

    normalized_macros = {}
    for old_name, value in raw_macros.items():
        normalized_name = name_map[old_name]
        normalized_value = normalize_macro_references(value, name_map) if isinstance(value, str) else value
        normalized_macros[normalized_name] = normalized_value

    raw_patterns: dict[str, str] = raw_data.get("model_patterns", {})
    normalized_patterns = {}
    for pattern, macro_name in raw_patterns.items():
        normalized_patterns[pattern] = (
            normalize_macro_references(macro_name, name_map)
            if "${" in macro_name
            else name_map.get(macro_name, normalize_macro_name(macro_name))
        )

    raw_variants: list[dict[str, str]] = raw_data.get("variants", [])
    normalized_variants = []
    for variant in raw_variants:
        normalized_variant = dict(variant)
        macro_name = normalized_variant.get("macro")
        if isinstance(macro_name, str):
            normalized_variant["macro"] = (
                normalize_macro_references(macro_name, name_map)
                if "${" in macro_name
                else name_map.get(macro_name, normalize_macro_name(macro_name))
            )
        normalized_variants.append(normalized_variant)

    return MacroConfig(
        macros=normalized_macros,
        model_patterns=normalized_patterns,
        variants=normalized_variants,
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
