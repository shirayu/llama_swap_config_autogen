"""YAML configuration generation logic."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from .config import load_macro_config
from .models import Config, MacroConfig, MultilineLiteral, Settings, YamlModelConfig


def extract_model_name_from_url(url: str, fallback_name: str) -> str:
    """Extract model name from HuggingFace URL or return fallback name."""
    if not url.startswith("https://huggingface.co/"):
        if not fallback_name:
            raise ValueError(f"No fallback name provided for non-HuggingFace URL: {url}")
        return fallback_name

    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid HuggingFace URL structure: {url}")

    user, repo = path_parts[0], path_parts[1].replace("-gguf", "").replace("-GGUF", "")
    filename = Path(url).stem
    parts = filename.split("-")
    fmt = parts[-4] if len(parts) >= 4 and parts[-2] == "of" else parts[-1]

    if not user or not repo or not fmt:
        raise ValueError(f"Unable to extract user, repo, or format from URL: {url}")

    return f"{user}/{repo}:{fmt}"


def generate_simple_id(url: str, fallback_name: str) -> str:
    """Generate simple ID directly from URL (without user, fmt prefix)"""
    if not url.startswith("https://huggingface.co/"):
        if not fallback_name:
            raise ValueError(f"No fallback name provided for non-HuggingFace URL: {url}")
        return fallback_name.replace(":", "-").lower()

    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid HuggingFace URL structure: {url}")

    repo = path_parts[1].replace("-gguf", "").replace("-GGUF", "")  # Remove user, use repo only

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

        for path_model_json in sorted(models_dir.glob("*.gguf.json")):
            prefix = ".".join(path_model_json.name.split(".")[:-2])

            with path_model_json.open() as inf:
                info = json.load(inf)
                url = info.get("url", "")
                display_name = generate_simple_id(url, prefix)
                model_id = extract_model_name_from_url(url, prefix)

            if model_id in ids:
                continue
            ids.add(model_id)

            path_model = path_model_json.parent.joinpath(prefix + ".gguf")
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


def extract_used_macros_from_commands(commands: list[str], all_macros: dict[str, str]) -> dict[str, str]:
    """Recursively extract macros and their dependencies used in commands"""
    used_macros = {}
    to_process = set()

    # Extract macros directly used in commands
    for command in commands:
        macro_matches = re.findall(r"\$\{([^}]+)\}", command)
        to_process.update(macro_matches)

    # Recursively resolve macro dependencies
    while to_process:
        current_macro = to_process.pop()
        if current_macro in used_macros or current_macro not in all_macros:
            continue

        # Add current macro to used list
        macro_value = all_macros[current_macro]
        used_macros[current_macro] = macro_value

        # Check if this macro references other macros
        nested_macros = re.findall(r"\$\{([^}]+)\}", macro_value)
        to_process.update(nested_macros)

    return used_macros


def generate_full_config(settings: Settings, config: Config) -> dict:
    """Generate complete configuration in llama-swap format"""
    models = generate_model_configs(settings)

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
    if config.macros and all_commands:
        used_macros = extract_used_macros_from_commands(all_commands, config.macros)
        if used_macros:
            output_config["macros"] = used_macros

    return output_config
