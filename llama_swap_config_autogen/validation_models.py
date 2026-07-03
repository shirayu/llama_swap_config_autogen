"""Pydantic models for validating llama-swap YAML configuration (business logic validation)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Individual model configuration (only fields needed for business logic validation)."""

    model_config = ConfigDict(extra="ignore")

    cmd: str
    cmdStop: str | None = ""
    proxy: str | None = "http://localhost:${PORT}"
    aliases: list[str] | None = Field(default_factory=list)
    checkEndpoint: str | None = "/health"


class GroupConfig(BaseModel):
    """Group configuration."""

    model_config = ConfigDict(extra="ignore")

    members: list[str]

    @field_validator("members")
    @classmethod
    def validate_members(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Group must have at least one member")
        if len(v) != len(set(v)):
            raise ValueError("Group members must be unique")
        return v


class HookOnStartup(BaseModel):
    """Startup hook configuration."""

    model_config = ConfigDict(extra="ignore")

    preload: list[str] | None = Field(default_factory=list)


class HooksConfig(BaseModel):
    """Hooks configuration."""

    model_config = ConfigDict(extra="ignore")

    on_startup: HookOnStartup | None = Field(default_factory=HookOnStartup)


class LlamaSwapConfig(BaseModel):
    """Main llama-swap configuration (only fields needed for business logic validation)."""

    model_config = ConfigDict(extra="ignore")

    startPort: int | None = Field(default=5800, gt=0, lt=65536)
    models: dict[str, ModelConfig]
    groups: dict[str, GroupConfig] | None = Field(default_factory=dict)
    macros: dict[str, str] | None = Field(default_factory=dict)
    hooks: HooksConfig | None = Field(default_factory=HooksConfig)

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        if not v:
            raise ValueError("At least one model must be defined")
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
