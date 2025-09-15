"""Data models for llama-swap configuration generation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    models: list[Path] = Field(description="List of model directory paths")
    macros: dict[str, str] = Field(default_factory=dict)
    model_patterns: dict[str, str] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(default_factory=list)
    default_ttl: int = Field(default=300)
    health_check_timeout: int = Field(default=240)
    log_level: str = Field(default="info")
    start_port: int = Field(default=9091)


class MacroConfig(BaseModel):
    macros: dict[str, str] = Field(default_factory=dict)
    model_patterns: dict[str, str] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(default_factory=list)


class Settings(BaseModel):
    models_dirs: list[Path]
    default_ttl: int = Field(default=300)
    health_check_timeout: int = Field(default=240)
    log_level: str = Field(default="info")
    start_port: int = Field(default=9091)
    config_file: Path


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
