"""YAML validation logic for llama-swap configurations."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from .validation_models import LlamaSwapConfig


class ValidationResult:
    """Result of YAML validation."""

    def __init__(self, is_valid: bool, errors: list[str] | None = None, warnings: list[str] | None = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)

    def format_report(self) -> str:
        """Format validation result as a human-readable report."""
        lines = []

        if self.is_valid:
            lines.append("✅ YAML validation passed")
        else:
            lines.append("❌ YAML validation failed")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")

        return "\n".join(lines)


def validate_yaml_syntax(yaml_path: Path) -> tuple[dict, list[str]]:
    """Validate YAML syntax and return parsed data and errors."""
    errors = []

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            errors.append("YAML file is empty or contains only null values")
            return {}, errors

        if not isinstance(data, dict):
            errors.append("YAML root must be a dictionary/object")
            return {}, errors

        return data, errors

    except yaml.YAMLError as e:
        errors.append(f"YAML syntax error: {e}")
        return {}, errors

    except FileNotFoundError:
        errors.append(f"File not found: {yaml_path}")
        return {}, errors

    except Exception as e:
        errors.append(f"Unexpected error reading file: {e}")
        return {}, errors


def validate_with_pydantic(data: dict, lenient: bool = False) -> ValidationResult:
    """Validate configuration data using Pydantic models."""
    result = ValidationResult(True)

    try:
        # Validate using Pydantic model
        config = LlamaSwapConfig(**data)

        # Additional business logic validations
        _validate_port_consistency(config, result)
        _validate_macro_circular_references(config, result)

        return result

    except ValidationError as e:
        result.is_valid = False

        for error in e.errors():
            field_path = " → ".join(str(loc) for loc in error["loc"]) if error["loc"] else "root"
            error_msg = error["msg"]

            # Clean up error message - remove long input values for readability
            if "Value error, " in error_msg:
                error_msg = error_msg.replace("Value error, ", "")

            if field_path == "root" or field_path == "":
                result.add_error(error_msg)
            else:
                result.add_error(f"Field '{field_path}': {error_msg}")

        return result

    except Exception as e:
        result.add_error(f"Unexpected validation error: {e}")
        return result


def _validate_port_consistency(config: LlamaSwapConfig, result: ValidationResult):
    """Validate PORT macro usage consistency."""
    for model_id, model_config in config.models.items():
        cmd_uses_port = "${PORT}" in model_config.cmd
        proxy_uses_port = config.models[model_id].proxy and "${PORT}" in (config.models[model_id].proxy or "")

        if proxy_uses_port and not cmd_uses_port:
            result.add_error(
                f"Model '{model_id}': proxy uses ${{PORT}} but cmd does not - "
                f"${{PORT}} is only available when used in cmd"
            )


def _validate_macro_circular_references(config: LlamaSwapConfig, result: ValidationResult):
    """Detect circular references in macros."""
    if not config.macros:
        return

    import re

    macro_pattern = re.compile(r"\$\{([a-zA-Z0-9_-]+)\}")

    def find_dependencies(macro_name: str, visited: set) -> bool:
        if macro_name in visited:
            return True  # Circular reference detected

        if not config.macros or macro_name not in config.macros:
            return False  # Not a defined macro (could be runtime macro like PORT)

        visited.add(macro_name)
        macro_value = config.macros[macro_name]
        dependencies = macro_pattern.findall(macro_value)

        for dep in dependencies:
            if find_dependencies(dep, visited.copy()):
                return True

        return False

    for macro_name in config.macros:
        if find_dependencies(macro_name, set()):
            result.add_error(f"Circular reference detected in macro '{macro_name}'")


def validate_yaml_file(yaml_path: Path, lenient: bool = False) -> ValidationResult:
    """Validate a YAML configuration file."""
    # Step 1: Validate YAML syntax
    data, syntax_errors = validate_yaml_syntax(yaml_path)

    if syntax_errors:
        result = ValidationResult(False)
        for error in syntax_errors:
            result.add_error(error)
        return result

    # Step 2: Validate with Pydantic models
    return validate_with_pydantic(data, lenient)


def validate_yaml_string(yaml_content: str) -> ValidationResult:
    """Validate YAML content from a string."""
    try:
        data = yaml.safe_load(yaml_content)

        if data is None:
            result = ValidationResult(False)
            result.add_error("YAML content is empty or contains only null values")
            return result

        if not isinstance(data, dict):
            result = ValidationResult(False)
            result.add_error("YAML root must be a dictionary/object")
            return result

        return validate_with_pydantic(data)

    except yaml.YAMLError as e:
        result = ValidationResult(False)
        result.add_error(f"YAML syntax error: {e}")
        return result

    except Exception as e:
        result = ValidationResult(False)
        result.add_error(f"Unexpected error: {e}")
        return result
