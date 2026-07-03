"""Tests for macro expansion functionality"""

import tempfile
from pathlib import Path

import pytest
import yaml

from llama_swap_config_autogen.config import load_macro_config
from llama_swap_config_autogen.generator import expand_macro, extract_used_macros_from_commands


class TestMacroExpansion:
    """Test macro expansion logic"""

    def test_simple_macro_expansion(self):
        """Test expanding a macro with no nested references"""
        macros = {"simple": "value"}
        result = expand_macro("simple", macros)
        assert result == "value"

    def test_single_level_nested_macro(self):
        """Test expanding a macro with one level of nesting"""
        macros = {"inner": "inner-value", "outer": "prefix ${inner} suffix"}
        result = expand_macro("outer", macros)
        assert result == "prefix inner-value suffix"

    def test_multi_level_nested_macro(self):
        """Test expanding a macro with multiple levels of nesting"""
        macros = {
            "level1": "L1",
            "level2": "${level1}-L2",
            "level3": "${level2}-L3",
        }
        result = expand_macro("level3", macros)
        assert result == "L1-L2-L3"

    def test_multiple_macro_references(self):
        """Test expanding a macro with multiple macro references"""
        macros = {
            "a": "valueA",
            "b": "valueB",
            "c": "valueC",
            "combined": "${a} ${b} ${c}",
        }
        result = expand_macro("combined", macros)
        assert result == "valueA valueB valueC"

    def test_complex_nested_macros(self):
        """Test real-world scenario with complex nesting"""
        macros = {
            "layers-full": "--n-gpu-layers 999",
            "context-large": "--ctx-size 65536",
            "layers-default": "${layers-full}",
            "context-default": "${context-large}",
            "default-params": "--log-disable --jinja ${layers-default} ${context-default}",
            "low-memory": "--no-mmap --mlock",
            "memory-efficient": "--cache-type-k q4_0 --cache-type-v q4_0",
            "memory-constrained-params": "${default-params} ${low-memory} ${memory-efficient}",
        }
        result = expand_macro("memory-constrained-params", macros)
        # With deduplication, parameter-value pairs are preserved
        expected = (
            "--log-disable --jinja --n-gpu-layers 999 "
            "--ctx-size 65536 --no-mmap --mlock --cache-type-k q4_0 --cache-type-v q4_0"
        )
        assert result == expected

    def test_circular_reference_detection(self):
        """Test that circular references are detected"""
        macros = {
            "a": "${b}",
            "b": "${a}",
        }
        with pytest.raises(ValueError, match="Circular macro reference"):
            expand_macro("a", macros)

    def test_builtin_macro_preservation(self):
        """Test that built-in macros like ${PORT} are preserved"""
        macros = {"custom": "value --port ${PORT}"}
        result = expand_macro("custom", macros)
        assert result == "value --port ${PORT}"

    def test_undefined_macro_in_nested_reference(self):
        """Test handling of undefined macros in nested references"""
        macros = {"outer": "prefix ${undefined} suffix"}
        result = expand_macro("outer", macros)
        # Undefined macros should be preserved as-is
        assert result == "prefix ${undefined} suffix"


class TestExtractUsedMacros:
    """Test macro extraction from commands"""

    def test_extract_simple_macros(self):
        """Test extracting simple macros from commands"""
        commands = ["command ${macro1}", "another ${macro2}"]
        macros = {"macro1": "value1", "macro2": "value2"}
        result = extract_used_macros_from_commands(commands, macros)
        assert result == {"macro1": "value1", "macro2": "value2"}

    def test_extract_and_expand_nested_macros(self):
        """Test that nested macros are expanded when extracted"""
        commands = ["command ${outer}"]
        macros = {"inner": "inner-value", "outer": "prefix ${inner} suffix"}
        result = extract_used_macros_from_commands(commands, macros)
        # The outer macro should be expanded to include inner's value
        assert result == {"outer": "prefix inner-value suffix"}

    def test_extract_complex_nested_macros(self):
        """Test extraction with complex nesting like in config.yaml"""
        commands = [
            "${binary} -m model.gguf --port ${PORT} --host 0.0.0.0 ${memory-constrained-params}",
        ]
        macros = {
            "binary": "/opt/llama.cpp/bin/llama-server",
            "layers-full": "--n-gpu-layers 999",
            "context-large": "--ctx-size 65536",
            "layers-default": "${layers-full}",
            "context-default": "${context-large}",
            "default-params": "--log-disable ${layers-default} ${context-default}",
            "low-memory": "--no-mmap --mlock",
            "memory-efficient": "--cache-type-k q4_0",
            "memory-constrained-params": "${default-params} ${low-memory} ${memory-efficient}",
        }
        result = extract_used_macros_from_commands(commands, macros)

        # binary should be kept as-is
        assert result["binary"] == "/opt/llama.cpp/bin/llama-server"

        # memory-constrained-params should be fully expanded with deduplication
        expected = "--log-disable --n-gpu-layers 999 --ctx-size 65536 --no-mmap --mlock --cache-type-k q4_0"
        assert result["memory-constrained-params"] == expected

    def test_builtin_macros_not_extracted(self):
        """Test that built-in macros like ${PORT} are not extracted"""
        commands = ["command ${PORT}"]
        macros = {}
        result = extract_used_macros_from_commands(commands, macros)
        # PORT is not in macros, so it should not be in result
        assert "PORT" not in result

    def test_circular_reference_handling(self):
        """Test that circular references are handled gracefully"""
        commands = ["command ${circular}"]
        macros = {"circular": "${circular}"}
        result = extract_used_macros_from_commands(commands, macros)
        # Should keep original due to circular reference detection
        assert result["circular"] == "${circular}"


class TestParameterDeduplication:
    """Test parameter deduplication logic"""

    def test_simple_deduplication(self):
        """Test removing duplicate parameters"""
        macros = {
            "a": "--flag value",
            "b": "--flag other",
            "combined": "${a} ${b}",
        }
        result = expand_macro("combined", macros)
        # --flag should appear only once (the last occurrence with its value)
        # Parameter-value pairs are kept together
        assert result == "--flag other"

    def test_deduplication_keeps_last_occurrence(self):
        """Test that the rightmost occurrence is kept"""
        macros = {
            "first": "--ctx-size 32768",
            "second": "--ctx-size 65536",
            "combined": "${first} ${second}",
        }
        result = expand_macro("combined", macros)
        # Should keep the last --ctx-size with its value (65536)
        assert result == "--ctx-size 65536"

    def test_real_world_deduplication(self):
        """Test deduplication with real config.yaml patterns"""
        macros = {
            "layers-full": "--n-gpu-layers 999",
            "context-large": "--ctx-size 65536",
            "layers-default": "${layers-full}",
            "context-default": "${context-large}",
            "default-params": "--log-disable --cache-type-k q8_0 "
            "--cache-type-v q8_0 ${layers-default} ${context-default}",
            "low-memory": "--no-mmap --mlock",
            "memory-efficient": "--cache-type-k q4_0 --cache-type-v q4_0",
            "memory-constrained-params": "${default-params} ${low-memory} ${memory-efficient}",
        }
        result = expand_macro("memory-constrained-params", macros)

        # Check that --cache-type-k appears only once (last occurrence with q4_0)
        assert result.count("--cache-type-k") == 1
        assert "--cache-type-k q4_0" in result
        # q8_0 should be removed (replaced by q4_0)
        assert result.count("q8_0") == 0
        # Check that q4_0 appears twice (once for each cache type)
        assert result.count("q4_0") == 2

        # Check that --cache-type-v appears only once (last occurrence with q4_0)
        assert result.count("--cache-type-v") == 1
        assert "--cache-type-v q4_0" in result

    def test_deduplication_preserves_order(self):
        """Test that deduplication preserves the order of first appearances"""
        macros = {
            "a": "--flag1 --flag2 --flag3",
            "b": "--flag2 --flag4",
            "combined": "${a} ${b}",
        }
        result = expand_macro("combined", macros)
        tokens = result.split()

        # --flag1 should appear before --flag3 (from a)
        assert tokens.index("--flag1") < tokens.index("--flag3")
        # --flag3 should appear before --flag4 (a before b)
        assert tokens.index("--flag3") < tokens.index("--flag4")
        # --flag2 should appear only once, at its last position (from b)
        assert result.count("--flag2") == 1


class TestMacroNameNormalization:
    """Test macro name normalization for llama-swap compatibility."""

    def test_dot_in_macro_name_is_normalized(self):
        config_data = {
            "models": ["/tmp/models"],
            "macros": {
                "glm.4.7-params": "--temperature 1.0",
                "glm-wrapper": "${glm.4.7-params} --repeat-penalty 1.0",
            },
            "model_patterns": {"glm-4.7-flash": "glm.4.7-params"},
            "variants": [
                {
                    "base_pattern": "glm-4.7-flash",
                    "suffix": " (thinking off)",
                    "macro": "${glm.4.7-params} --reasoning off",
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = Path(f.name)

        try:
            macro_config = load_macro_config(temp_path)
            assert "glm-4-7-params" in macro_config.macros
            assert macro_config.macros["glm-wrapper"] == "${glm-4-7-params} --repeat-penalty 1.0"
            assert macro_config.model_patterns["glm-4.7-flash"].macro == "glm-4-7-params"
            assert macro_config.model_patterns["glm-4.7-flash"].emit_base is True
            assert macro_config.variants[0]["macro"] == "${glm-4-7-params} --reasoning off"
        finally:
            temp_path.unlink()

    def test_model_pattern_object_config_is_normalized(self):
        config_data = {
            "models": ["/tmp/models"],
            "macros": {
                "gemma.4.params": "--ctx-size 32768",
            },
            "model_patterns": {
                "gemma-4-": {
                    "macro": "gemma.4.params",
                    "emit_base": False,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = Path(f.name)

        try:
            macro_config = load_macro_config(temp_path)
            assert macro_config.model_patterns["gemma-4-"].macro == "gemma-4-params"
            assert macro_config.model_patterns["gemma-4-"].emit_base is False
        finally:
            temp_path.unlink()


class TestVariantPresets:
    """Test variant presets and template argument binding functionality"""

    def test_variant_presets_normalization(self):
        """Test loading variant presets and macro normalization"""
        config_data = {
            "models": ["/tmp/models"],
            "macros": {
                "qwen.cpu-params": "--threads 16",
            },
            "variant_presets": {
                "cpu.variant": [
                    {
                        "suffix": " (with CPU)",
                        "macro": "${cpu.macro}",
                    }
                ]
            },
            "model_patterns": {
                "qwen3": {
                    "macro": "default-params",
                    "cpu.macro": "qwen.cpu-params",
                    "variants": ["cpu.variant"],
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = Path(f.name)

        try:
            macro_config = load_macro_config(temp_path)
            # variant_presets keys and inner macro references should be normalized
            assert "cpu-variant" in macro_config.variant_presets
            preset_items = macro_config.variant_presets["cpu-variant"]
            assert len(preset_items) == 1
            assert preset_items[0].suffix == " (with CPU)"
            assert preset_items[0].macro == "${cpu-macro}"

            # model_pattern's variants list should be normalized
            pattern_config = macro_config.model_patterns["qwen3"]
            assert pattern_config.variants == ["cpu-variant"]

            # custom extra fields (arguments) should be preserved in model_extra
            assert pattern_config.model_extra is not None
            assert pattern_config.model_extra["cpu-macro"] == "qwen-cpu-params"
        finally:
            temp_path.unlink()

    def test_resolve_variant_macro_template(self):
        """Test resolving templates like ${cpu-macro} with custom pattern config arguments"""
        from llama_swap_config_autogen.generator import resolve_variant_macro_template
        from llama_swap_config_autogen.models import ModelPatternConfig

        pattern_config = ModelPatternConfig.model_validate({"macro": "default-params", "cpu-macro": "qwen-cpu-params"})

        template = "some prefix ${cpu-macro} and global ${global-macro}"
        resolved = resolve_variant_macro_template(template, pattern_config)

        # cpu-macro should be resolved, global-macro should be kept as-is
        assert resolved == "some prefix qwen-cpu-params and global ${global-macro}"


class TestParameterizedMacros:
    """Test parameterized macro expansion (arguments with positional parameters)"""

    def test_parameterized_macro_expansion(self):
        """Test expanding macros with positional parameters ${1}, ${2}"""
        macros = {
            "ctx": "--ctx-size ${1}",
            "ngl": "--n-gpu-layers ${1}",
            "cpu-offload": "--n-cpu-moe ${1} --threads ${2}",
            "default-params": "--common ${ngl:999} ${ctx:32768}",
            "cpu-default": "${cpu-offload:12,16}",
        }

        # Simple single parameter expansion
        assert expand_macro("ctx:4096", macros) == "--ctx-size 4096"
        assert expand_macro("ngl:15", macros) == "--n-gpu-layers 15"

        # Multi-parameter expansion
        assert expand_macro("cpu-offload:12,16", macros) == "--n-cpu-moe 12 --threads 16"

        # Nested parameterized macro expansion
        assert expand_macro("default-params", macros) == "--common --n-gpu-layers 999 --ctx-size 32768"
        assert expand_macro("cpu-default", macros) == "--n-cpu-moe 12 --threads 16"

    def test_parameterized_macro_normalization(self):
        """Test normalization of macro references with arguments"""
        from llama_swap_config_autogen.config import normalize_macro_references

        name_map = {"cpu-offload": "cpu-offload"}

        # Dot in parameter value should be preserved, macro name normalized
        expr = "${cpu.offload:12,16} and ${some.macro:3.5}"
        name_map_extended = {"cpu-offload": "cpu-offload", "some-macro": "some-macro"}
        normalized = normalize_macro_references(expr, name_map_extended)
        assert normalized == "${cpu-offload:12,16} and ${some-macro:3.5}"

    def test_extract_parameterized_macros(self):
        """Test extracting parameterized macros from commands"""
        commands = [
            "command ${cpu-offload:12,16}",
        ]
        macros = {
            "cpu-offload": "--n-cpu-moe ${1} --threads ${2}",
        }
        result = extract_used_macros_from_commands(commands, macros)
        assert "cpu-offload:12,16" in result
        assert result["cpu-offload:12,16"] == "--n-cpu-moe 12 --threads 16"
