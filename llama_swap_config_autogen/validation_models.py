"""Pydantic models for validating llama-swap YAML configuration."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelFilters(BaseModel):
    """Model filters configuration."""

    strip_params: str | None = Field(default="", description="Comma-separated parameters to strip")

    @field_validator("strip_params")
    @classmethod
    def validate_strip_params(cls, v: str | None) -> str:
        if v is None:
            return ""
        # Basic validation: ensure it's comma-separated
        if v and not re.match(r"^[a-zA-Z0-9_,\s]+$", v):
            raise ValueError("strip_params must contain only alphanumeric characters, underscores, commas, and spaces")
        return v


class ModelConfig(BaseModel):
    """Individual model configuration."""

    cmd: str = Field(..., description="Command to start the model")
    cmdStop: str | None = Field(default="", description="Command to stop the model")
    proxy: str | None = Field(default="http://localhost:${PORT}", description="Proxy URL")
    aliases: list[str] | None = Field(default_factory=list, description="Model aliases")
    env: list[str] | None = Field(default_factory=list, description="Environment variables")
    checkEndpoint: str | None = Field(default="/health", description="Health check endpoint")
    ttl: int | None = Field(default=0, ge=0, description="Time to live in seconds")
    unlisted: bool | None = Field(default=False, description="Whether model is unlisted")
    useModelName: str | None = Field(default="", description="Override model name sent to upstream")
    name: str | None = Field(default="", description="Display name")
    description: str | None = Field(default="", description="Model description")
    concurrencyLimit: int | None = Field(default=0, ge=0, description="Concurrency limit")
    filters: ModelFilters | None = Field(default_factory=ModelFilters, description="Model filters")

    @field_validator("cmd")
    @classmethod
    def validate_cmd(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("cmd cannot be empty")
        return v

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: list[str] | None) -> list[str]:
        if v is None:
            return []
        for env_var in v:
            if "=" not in env_var:
                raise ValueError(f"Environment variable must be in format KEY=value, got: {env_var}")
        return v

    @field_validator("proxy")
    @classmethod
    def validate_proxy(cls, v: str | None) -> str:
        if v is None:
            return "http://localhost:${PORT}"
        if v and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("proxy must start with http:// or https://")
        return v


class GroupConfig(BaseModel):
    """Group configuration."""

    swap: bool | None = Field(default=True, description="Enable model swapping within group")
    exclusive: bool | None = Field(default=True, description="Group is exclusive")
    persistent: bool | None = Field(default=False, description="Group is persistent")
    members: list[str] = Field(..., description="Group member model IDs")

    @field_validator("members")
    @classmethod
    def validate_members(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Group must have at least one member")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Group members must be unique")
        return v


class HookOnStartup(BaseModel):
    """Startup hook configuration."""

    preload: list[str] | None = Field(default_factory=list, description="Models to preload on startup")


class HooksConfig(BaseModel):
    """Hooks configuration."""

    on_startup: HookOnStartup | None = Field(default_factory=HookOnStartup, description="Startup hooks")


class LlamaSwapConfig(BaseModel):
    """Main llama-swap configuration."""

    healthCheckTimeout: int | None = Field(default=120, ge=15, description="Health check timeout in seconds")
    logRequests: bool | None = Field(default=False, description="Log requests")
    logLevel: str | None = Field(default="info", description="Log level")
    metricsMaxInMemory: int | None = Field(default=1000, ge=0, description="Max metrics in memory")
    models: dict[str, ModelConfig] = Field(..., description="Model configurations")
    profiles: dict[str, list[str]] | None = Field(default_factory=dict, description="Profile configurations")
    groups: dict[str, GroupConfig] | None = Field(default_factory=dict, description="Group configurations")
    macros: dict[str, str] | None = Field(default_factory=dict, description="Macro definitions")
    startPort: int | None = Field(default=5800, gt=0, lt=65536, description="Starting port number")
    hooks: HooksConfig | None = Field(default_factory=HooksConfig, description="Hook configurations")

    @field_validator("logLevel")
    @classmethod
    def validate_log_level(cls, v: str | None) -> str:
        if v is None:
            return "info"
        valid_levels = {"debug", "info", "warn", "error"}
        if v not in valid_levels:
            raise ValueError(f"logLevel must be one of {valid_levels}, got: {v}")
        return v

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        if not v:
            raise ValueError("At least one model must be defined")
        return v

    @field_validator("macros")
    @classmethod
    def validate_macros(cls, v: dict[str, str] | None) -> dict[str, str]:
        if v is None:
            return {}

        # Validate macro names
        reserved_names = {"PORT", "MODEL_ID", "PID"}
        macro_name_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

        for name, value in v.items():
            if len(name) >= 64:
                raise ValueError(f"Macro name '{name}' exceeds maximum length of 63 characters")
            if not macro_name_pattern.match(name):
                raise ValueError(f"Macro name '{name}' contains invalid characters")
            if name in reserved_names:
                raise ValueError(f"Macro name '{name}' is reserved")
            if len(value) >= 1024:
                raise ValueError(f"Macro value for '{name}' exceeds maximum length of 1024 characters")

        return v

    @model_validator(mode="after")
    def validate_config_consistency(self) -> LlamaSwapConfig:
        # Validate aliases are unique across all models
        aliases_to_model = {}
        for model_id, model_config in self.models.items():
            if model_config.aliases:
                for alias in model_config.aliases:
                    if alias in aliases_to_model:
                        raise ValueError(
                            f"Duplicate alias '{alias}' found in models: {aliases_to_model[alias]} and {model_id}"
                        )
                    aliases_to_model[alias] = model_id

        # Validate group members exist in models
        if self.groups:
            all_members = set()
            undefined_models = []
            for group_id, group_config in self.groups.items():
                for member in group_config.members:
                    if member not in self.models:
                        undefined_models.append(f"Group '{group_id}' references undefined model '{member}'")
                    if member in all_members:
                        raise ValueError(f"Model '{member}' is referenced in multiple groups")
                    all_members.add(member)

            # Raise all undefined model errors at once
            if undefined_models:
                error_msg = "Undefined model references found:\n" + "\n".join(
                    f"  - {error}" for error in undefined_models
                )
                raise ValueError(error_msg)

        # Validate hooks preload models exist
        if self.hooks and self.hooks.on_startup and self.hooks.on_startup.preload:
            for model_id in self.hooks.on_startup.preload:
                if model_id not in self.models and model_id not in aliases_to_model:
                    raise ValueError(f"Hooks preload references unknown model '{model_id}'")

        # Basic macro usage validation (detect obvious undefined macros and nested macros)
        if self.macros:
            macro_pattern = re.compile(r"\$\{([a-zA-Z0-9_-]+)\}")
            nested_macro_pattern = re.compile(r"\$\{\$\{")  # Detect nested ${${...} patterns

            for model_id, model_config in self.models.items():
                for field_name, field_value in [
                    ("cmd", model_config.cmd),
                    ("cmdStop", model_config.cmdStop or ""),
                    ("proxy", model_config.proxy or ""),
                    ("checkEndpoint", model_config.checkEndpoint or ""),
                ]:
                    # Check for nested macros
                    if nested_macro_pattern.search(field_value):
                        raise ValueError(
                            f"Nested macro detected in model '{model_id}' field '{field_name}': {field_value}"
                        )

                    matches = macro_pattern.findall(field_value)
                    for macro_name in matches:
                        # Skip runtime macros
                        if macro_name in {"PORT", "MODEL_ID", "PID"}:
                            continue
                        if macro_name not in self.macros:
                            raise ValueError(
                                f"Unknown macro '${{{macro_name}}}' in model '{model_id}' field '{field_name}'"
                            )

        return self
