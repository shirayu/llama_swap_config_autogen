"""Tests for YAML validation functionality."""

import tempfile
from pathlib import Path

import pytest
import yaml

from llama_swap_config_autogen.validator import ValidationResult, validate_yaml_file, validate_yaml_string


class TestValidationResult:
    """Test ValidationResult class."""

    def test_valid_result(self):
        result = ValidationResult(True)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result(self):
        result = ValidationResult(False, ["error1"], ["warning1"])
        assert not result.is_valid
        assert result.errors == ["error1"]
        assert result.warnings == ["warning1"]

    def test_add_error(self):
        result = ValidationResult(True)
        result.add_error("test error")
        assert not result.is_valid
        assert "test error" in result.errors

    def test_add_warning(self):
        result = ValidationResult(True)
        result.add_warning("test warning")
        assert result.is_valid  # warnings don't change validity
        assert "test warning" in result.warnings

    def test_format_report_valid(self):
        result = ValidationResult(True)
        report = result.format_report()
        assert "✅ YAML validation passed" in report

    def test_format_report_invalid(self):
        result = ValidationResult(False, ["error1", "error2"], ["warning1"])
        report = result.format_report()
        assert "❌ YAML validation failed" in report
        assert "error1" in report
        assert "error2" in report
        assert "warning1" in report


class TestYamlStringValidation:
    """Test YAML string validation."""

    def test_valid_yaml_string(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: "llama-server --port ${PORT}"
    name: "Test Model"
macros:
  test-macro: "test value"
"""
        result = validate_yaml_string(yaml_content)
        assert result.is_valid

    def test_invalid_yaml_syntax(self):
        yaml_content = """
healthCheckTimeout: 120
invalid: [unclosed list
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("syntax error" in error.lower() for error in result.errors)

    def test_empty_yaml(self):
        result = validate_yaml_string("")
        assert not result.is_valid
        assert any("empty" in error.lower() for error in result.errors)

    def test_null_yaml(self):
        result = validate_yaml_string("null")
        assert not result.is_valid
        assert any("empty" in error.lower() for error in result.errors)

    def test_non_dict_yaml(self):
        result = validate_yaml_string("- item1\n- item2")
        assert not result.is_valid
        assert any("dictionary" in error.lower() for error in result.errors)


class TestYamlFileValidation:
    """Test YAML file validation."""

    def test_validate_reference_file(self):
        """Test validation of the reference YAML file."""
        reference_file = Path(__file__).parent / "example.llam_swap.original.yaml"
        if reference_file.exists():
            result = validate_yaml_file(reference_file)
            # Note: Reference file might have some validation issues, so we just check it doesn't crash
            assert isinstance(result, ValidationResult)

    def test_file_not_found(self):
        non_existent_file = Path("/non/existent/file.yaml")
        result = validate_yaml_file(non_existent_file)
        assert not result.is_valid
        assert any("not found" in error.lower() for error in result.errors)

    def test_valid_yaml_file(self):
        valid_yaml = {
            "healthCheckTimeout": 120,
            "logLevel": "info",
            "startPort": 5800,
            "models": {"test-model": {"cmd": "llama-server --port ${PORT}", "name": "Test Model"}},
            "macros": {"test-macro": "test value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_yaml, f)
            temp_path = Path(f.name)

        try:
            result = validate_yaml_file(temp_path)
            assert result.is_valid
        finally:
            temp_path.unlink()


class TestValidationLogic:
    """Test specific validation logic."""

    def test_missing_models(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("models" in error.lower() for error in result.errors)

    def test_empty_cmd(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: ""
    name: "Test Model"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("cmd" in error.lower() and "empty" in error.lower() for error in result.errors)

    def test_invalid_log_level(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: invalid
startPort: 5800
models:
  test-model:
    cmd: "llama-server"
    name: "Test Model"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("loglevel" in error.lower() for error in result.errors)

    def test_invalid_port_range(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 70000
models:
  test-model:
    cmd: "llama-server"
    name: "Test Model"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("startport" in error.lower() for error in result.errors)

    def test_duplicate_aliases(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  model1:
    cmd: "llama-server"
    aliases: ["alias1"]
  model2:
    cmd: "llama-server"
    aliases: ["alias1"]
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("duplicate alias" in error.lower() for error in result.errors)

    def test_invalid_env_format(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: "llama-server"
    env:
      - "VALID_VAR=value"
      - "INVALID_VAR_NO_EQUALS"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("environment" in error.lower() for error in result.errors)

    def test_port_consistency_validation(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: "llama-server"
    proxy: "http://localhost:${PORT}"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("port" in error.lower() and "cmd" in error.lower() for error in result.errors)

    def test_unknown_macro_reference(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: "llama-server ${UNDEFINED_MACRO}"
macros:
  defined_macro: "value"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("unknown macro" in error.lower() for error in result.errors)

    def test_reserved_macro_name(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: "llama-server"
macros:
  PORT: "invalid"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("reserved" in error.lower() for error in result.errors)

    def test_group_references_unknown_model(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: "llama-server"
groups:
  test-group:
    members: ["test-model", "unknown-model"]
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("undefined model" in error.lower() for error in result.errors)

    def test_nested_macro_detection(self):
        yaml_content = """
healthCheckTimeout: 120
logLevel: info
startPort: 5800
models:
  test-model:
    cmd: "llama-server ${${deepseek-r1-params} ${fast-inference}}"
macros:
  deepseek-r1-params: "value"
  fast-inference: "value"
"""
        result = validate_yaml_string(yaml_content)
        assert not result.is_valid
        assert any("nested macro" in error.lower() for error in result.errors)


if __name__ == "__main__":
    pytest.main([__file__])
