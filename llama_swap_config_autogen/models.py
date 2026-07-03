"""Data models for llama-swap configuration generation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MmprojConfig(BaseModel):
    enabled: bool = Field(default=True)
    auto_attach: bool = Field(default=True)
    arg: str = Field(default="--mmproj")
    overrides: dict[str, Path] = Field(default_factory=dict)
    generate_no_mmproj_variant: bool = Field(default=False)
    no_mmproj_suffix: str = Field(default=" (no mmproj)")


class ModelLabelRule(BaseModel):
    pattern: str
    label: str
    requires_mmproj: bool = Field(default=False)


class ModelLabelsConfig(BaseModel):
    mmproj_default: str = Field(default=" 🌐")
    rules: list[ModelLabelRule] = Field(default_factory=list)


class Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    models: list[Path] = Field(description="List of model directory paths")
    macros: dict[str, str] = Field(default_factory=dict)
    model_patterns: dict[str, Any] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(default_factory=list)
    variant_presets: dict[str, Any] = Field(default_factory=dict)
    mmproj: MmprojConfig = Field(default_factory=MmprojConfig)
    model_labels: ModelLabelsConfig = Field(default_factory=ModelLabelsConfig)
    default_ttl: int = Field(default=300)
    vram_estimation: bool = Field(default=False)


class VariantPresetItem(BaseModel):
    suffix: str
    macro: str


class ModelPatternConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    macro: str
    emit_base: bool = Field(default=True)
    variants: list[str] = Field(default_factory=list)


class MacroConfig(BaseModel):
    macros: dict[str, str] = Field(default_factory=dict)
    model_patterns: dict[str, ModelPatternConfig] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(default_factory=list)
    variant_presets: dict[str, list[VariantPresetItem]] = Field(default_factory=dict)


class Settings(BaseModel):
    models_dirs: list[Path]
    default_ttl: int = Field(default=300)
    config_file: Path
    vram_estimation: bool = Field(default=False)


class MultilineLiteral(str):
    """Custom YAML string class to force literal block scalar style"""

    pass


def multiline_literal_representer(dumper: yaml.Dumper, data: MultilineLiteral) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(MultilineLiteral, multiline_literal_representer)


class YamlModelConfig(BaseModel):
    ttl: int = Field(description="Time to live for the model in seconds")
    cmd: MultilineLiteral = Field(description="Command to run the model")
    name: str = Field(description="Display name for the model")

    model_config = {"arbitrary_types_allowed": True}


class YamlConfig(BaseModel):
    healthCheckTimeout: int = Field(description="Health check timeout in seconds")
    logLevel: str = Field(description="Log level")
    startPort: int = Field(description="Starting port number")
    models: dict[str, YamlModelConfig] = Field(default_factory=dict, description="Model configurations")
