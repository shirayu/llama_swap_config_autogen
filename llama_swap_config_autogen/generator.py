"""YAML configuration generation logic."""

import re

from .config import load_macro_config
from .models import Config, MacroConfig, MultilineLiteral, Settings, YamlModelConfig


def extract_model_name_from_url(filename: str) -> str:
    """Extract model name from HuggingFace URL or return fallback name."""

    frn = filename.split("_", maxsplit=2)
    if len(frn) < 3:
        raise ValueError(f"Invalid filename: {filename}")

    user, repo = frn[0], frn[1].replace("-gguf", "").replace("-GGUF", "")
    parts = frn[-1].replace(".gguf", "").split("-")
    fmt = parts[-4] if len(frn) >= 4 and parts[-2] == "of" else parts[-1]

    return f"{user}/{repo}:{fmt}"


def generate_simple_id(filename: str) -> str:
    """Generate simple ID directly from URL (without user, fmt prefix)"""

    frn = filename.split("_", maxsplit=2)[-1]
    repo = frn.replace(".gguf", "").replace(".GGUF", "")  # Remove user, use repo only

    return f"{repo}".lower()


def get_model_macro(model_name: str, macro_config: MacroConfig) -> str:
    """Get appropriate macro based on model name"""
    # Check model name patterns
    for pattern, macro_name in macro_config.model_patterns.items():
        if pattern in model_name:
            return macro_name

    # Use default parameters
    return "default-params"


def format_command_with_macro(model_path: str, macro_name: str) -> MultilineLiteral:
    """Generate command using macro (llama-swap format)"""
    # Check if macro_name already contains ${...} references (for complex macros)
    if macro_name.startswith("${") and macro_name.endswith("}"):
        # It's already a formatted macro expression, use as-is
        cmd = f"${{binary}} -m {model_path} --port ${{PORT}} --host 0.0.0.0 {macro_name}"
    else:
        # It's a simple macro name, wrap it with ${}
        cmd = f"${{binary}} -m {model_path} --port ${{PORT}} --host 0.0.0.0 ${{{macro_name}}}"
    return MultilineLiteral(cmd)


def generate_model_configs(settings: Settings) -> dict[str, YamlModelConfig]:
    # Load macro configuration
    macro_config = load_macro_config(settings.config_file)

    models = {}
    ids = set()

    for models_dir in settings.models_dirs:
        if not models_dir.exists():
            continue

        for path_model in sorted(models_dir.glob("*.gguf")):
            display_name = generate_simple_id(path_model.name)
            model_id = extract_model_name_from_url(path_model.name)

            if model_id in ids:
                continue
            ids.add(model_id)

            macro_name = get_model_macro(display_name, macro_config)

            cmd = format_command_with_macro(str(path_model), macro_name)

            models[model_id] = YamlModelConfig(ttl=settings.default_ttl, cmd=cmd, name=display_name)

            # Generate variant models
            for variant in macro_config.variants:
                base_pattern = variant.get("base_pattern", "")
                suffix = variant.get("suffix", "")
                variant_macro = variant.get("macro", "")

                if base_pattern.lower() in display_name.lower() and suffix and variant_macro:
                    # Generate variant_id in YAML key suitable format from model_id
                    cleaned_suffix = (
                        suffix.replace(" ", "-").replace("(", "").replace(")", "").replace("+", "plus").lower()
                    )
                    variant_id = f"{model_id}-{cleaned_suffix}"
                    variant_display_name = f"{display_name}{suffix}"
                    if variant_id not in models:  # Avoid duplicates
                        variant_cmd = format_command_with_macro(str(path_model), variant_macro)
                        models[variant_id] = YamlModelConfig(
                            ttl=settings.default_ttl, cmd=variant_cmd, name=variant_display_name
                        )

    if not models:
        raise ValueError("No models found. Please check your models directory and ensure .gguf.json files exist.")

    return models


def deduplicate_parameters(expanded_value: str) -> str:
    """Remove duplicate parameters, keeping the last occurrence (rightmost priority)

    Handles parameter-value pairs like --cache-type-k q8_0 correctly by tracking
    which parameter names (flags starting with -) appear and keeping only the last
    occurrence along with its value.
    """
    tokens = expanded_value.split()

    # Track parameters and their positions
    # Key: parameter name (flag), Value: list of indices (parameter index, value indices)
    param_occurrences = {}
    standalone_tokens = {}  # Tokens that are not part of parameter pairs

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check if this is a parameter flag (starts with -)
        if token.startswith("-"):
            # Collect any non-flag tokens that follow as values
            value_indices = []
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("-"):
                value_indices.append(j)
                j += 1

            # Store this parameter occurrence (will be overwritten if duplicate)
            param_occurrences[token] = (i, value_indices)
            i = j  # Skip to next parameter
        else:
            # This is a standalone value (shouldn't happen in well-formed params)
            standalone_tokens[i] = token
            i += 1

    # Rebuild the parameter string with deduplication
    # Collect all indices to include (parameters and their values)
    indices_to_include = set()

    # Add the last occurrence of each parameter and its values
    for param_idx, value_indices in param_occurrences.values():
        indices_to_include.add(param_idx)
        indices_to_include.update(value_indices)

    # Add standalone tokens
    indices_to_include.update(standalone_tokens.keys())

    # Rebuild the string in order
    result = [tokens[i] for i in sorted(indices_to_include)]
    return " ".join(result)


def expand_macro(macro_name: str, all_macros: dict[str, str], visited: set[str] | None = None) -> str:
    """Recursively expand a macro by resolving all nested macro references"""
    if visited is None:
        visited = set()

    if macro_name in visited:
        raise ValueError(f"Circular macro reference detected: {macro_name}")

    if macro_name not in all_macros:
        # Return as-is if macro not found (could be built-in like ${PORT})
        return f"${{{macro_name}}}"

    visited.add(macro_name)
    macro_value = all_macros[macro_name]

    # Find all nested macro references
    nested_macros = re.findall(r"\$\{([^}]+)\}", macro_value)

    # Expand each nested macro
    expanded_value = macro_value
    for nested_macro in nested_macros:
        if nested_macro in all_macros:
            # Recursively expand nested macro
            nested_expanded = expand_macro(nested_macro, all_macros, visited.copy())
            # Replace the macro reference with its expanded value
            expanded_value = expanded_value.replace(f"${{{nested_macro}}}", nested_expanded)

    # Deduplicate parameters, keeping the last (rightmost) occurrence
    expanded_value = deduplicate_parameters(expanded_value)

    return expanded_value


def extract_used_macros_from_commands(commands: list[str], all_macros: dict[str, str]) -> dict[str, str]:
    """Extract macros used in commands and expand nested macro references"""
    used_macros = {}
    to_process = set()

    # Extract macros directly used in commands
    for command in commands:
        macro_matches = re.findall(r"\$\{([^}]+)\}", command)
        to_process.update(macro_matches)

    # Process each macro and expand it
    for macro_name in to_process:
        if macro_name not in all_macros:
            continue

        # Expand the macro to resolve all nested references
        try:
            expanded_value = expand_macro(macro_name, all_macros)
            used_macros[macro_name] = expanded_value
        except ValueError as e:
            # If circular reference detected, keep original value
            print(f"Warning: {e}. Keeping original macro definition.")
            used_macros[macro_name] = all_macros[macro_name]

    return used_macros


def generate_full_config(settings: Settings, config: Config) -> dict:
    """Generate complete configuration in llama-swap format"""
    models = generate_model_configs(settings)
    macro_config = load_macro_config(settings.config_file)

    # Create llama-swap format configuration
    output_config = {
        "healthCheckTimeout": settings.health_check_timeout,
        "logLevel": settings.log_level,
        "startPort": settings.start_port,
    }

    # Add model configurations and collect commands simultaneously
    output_config["models"] = {}
    all_commands = []
    for model_id, model_config in models.items():
        output_config["models"][model_id] = {
            "ttl": model_config.ttl,
            "cmd": model_config.cmd,
            "name": model_config.name,
        }
        # Collect commands as strings
        cmd_str = str(model_config.cmd)
        all_commands.append(cmd_str)

    # Extract and add only used macros
    if macro_config.macros and all_commands:
        used_macros = extract_used_macros_from_commands(all_commands, macro_config.macros)
        if used_macros:
            output_config["macros"] = used_macros

    return output_config
