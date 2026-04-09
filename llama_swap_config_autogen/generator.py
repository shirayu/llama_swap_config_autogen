"""YAML configuration generation logic."""

import logging
import re
from pathlib import Path

from .config import load_macro_config
from .gguf_metadata import GGUFMetadataCache, estimate_vram_gb, get_gguf_metadata
from .models import Config, MacroConfig, MultilineLiteral, Settings, YamlModelConfig

logger = logging.getLogger(__name__)

MMPROJ_PATTERN = re.compile(r"mmproj", re.IGNORECASE)
BF16_PATTERN = re.compile(r"bf16", re.IGNORECASE)
QUANTIZATION_PATTERN = re.compile(r"-(Q\d(?:_[A-Z0-9]+)+|BF16|F16)(?=\.gguf$)", re.IGNORECASE)
NGL_PATTERN = re.compile(r"(?:-ngl|--n-gpu-layers)\s+(\d+)")
CONTEXT_PATTERN = re.compile(r"(?:-c|--ctx-size)\s+(\d+)")
CACHE_TYPE_K_PATTERN = re.compile(r"--cache-type-k\s+([^\s]+)")
CACHE_TYPE_V_PATTERN = re.compile(r"--cache-type-v\s+([^\s]+)")
CPU_OFFLOAD_PATTERN = re.compile(r"(?:--cpu-moe|--n-cpu-moe\b|-ot\b[^\n]*=CPU)", re.IGNORECASE)
N_CPU_MOE_PATTERN = re.compile(r"--n-cpu-moe\s+(\d+)")


def extract_quantization_suffix(filename: str) -> str:
    """Extract quantization suffix such as Q4_K_M, BF16, or F16 from the filename."""
    match = QUANTIZATION_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not determine quantization suffix from filename: {filename}")
    return match.group(1).upper()


def should_ignore_first_segment(models_dir: Path, model_files: list[Path]) -> bool:
    if not model_files:
        return False

    per_top_level: dict[str, set[int]] = {}
    for path in model_files:
        parts = path.relative_to(models_dir).parent.parts
        depth = len(parts)
        if depth not in {1, 2, 3}:
            raise ValueError(
                f"Unexpected model directory depth for '{path}'. "
                "Expected 'model/*.gguf' or 'model/variant/*.gguf', "
                "with an optional ignored leading family directory."
            )
        per_top_level.setdefault(parts[0], set()).add(depth)

    has_family_layout = False
    has_direct_layout = False
    has_ambiguous_layout = False

    for depths in per_top_level.values():
        if 3 in depths and 1 in depths:
            raise ValueError(
                f"Unexpected mixed model directory depths under '{models_dir}'. "
                "Use one consistent layout style per models directory."
            )
        if 3 in depths:
            has_family_layout = True
        elif 1 in depths:
            has_direct_layout = True
        else:
            has_ambiguous_layout = True

    if has_family_layout and has_direct_layout:
        raise ValueError(
            f"Unexpected mixed model directory depths under '{models_dir}'. "
            "Use one consistent layout style per models directory."
        )
    if has_family_layout and has_ambiguous_layout:
        raise ValueError(
            f"Unexpected mixed model directory depths under '{models_dir}'. "
            "Use one consistent layout style per models directory."
        )

    return has_family_layout


def build_display_name(models_dir: Path, model_path: Path, ignore_first_segment: bool) -> str:
    relative_parent = model_path.relative_to(models_dir).parent
    parts = relative_parent.parts[1:] if ignore_first_segment else relative_parent.parts
    depth = len(parts)
    if depth not in {1, 2}:
        raise ValueError(
            f"Unexpected model directory depth for '{model_path}'. "
            "Expected 'model/*.gguf' or 'model/variant/*.gguf', "
            "with an optional ignored leading family directory."
        )
    return "/".join(parts).lower()


def build_model_id(models_dir: Path, model_path: Path, ignore_first_segment: bool) -> str:
    display_name = build_display_name(models_dir, model_path, ignore_first_segment)
    quantization = extract_quantization_suffix(model_path.name)
    return f"{display_name}:{quantization}"


def build_model_name(display_name: str, quantization: str) -> str:
    return f"{display_name}:{quantization}"


def get_model_macro(model_name: str, macro_config: MacroConfig) -> str:
    """Get appropriate macro based on model name"""
    # Check model name patterns
    for pattern, macro_name in macro_config.model_patterns.items():
        if pattern.lower() in model_name.lower():
            return macro_name

    # Use default parameters
    return "default-params"


def is_mmproj_file(path_model: Path) -> bool:
    return bool(MMPROJ_PATTERN.search(path_model.name))


def select_mmproj_path_for_model(
    model_path: Path,
    model_id: str,
    display_name: str,
    mmproj_overrides: dict[str, Path],
    mmproj_by_prefix: dict[str, list[Path]],
    auto_attach: bool,
) -> Path | None:
    override = (
        mmproj_overrides.get(model_id) or mmproj_overrides.get(display_name) or mmproj_overrides.get(model_path.name)
    )
    if override:
        return override

    if not auto_attach:
        return None

    candidates = mmproj_by_prefix.get(str(model_path.parent), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        bf16_candidates = [candidate for candidate in candidates if BF16_PATTERN.search(candidate.name)]
        if len(bf16_candidates) == 1:
            return bf16_candidates[0]

    return None


def format_suffix_for_id(suffix: str) -> str:
    return suffix.replace(" ", "-").replace("(", "").replace(")", "").replace("+", "plus").lower()


def format_command_with_macro(
    model_path: str,
    macro_name: str,
    mmproj_path: str | None = None,
    mmproj_arg: str = "--mmproj",
) -> MultilineLiteral:
    """Generate command using macro (llama-swap format)"""
    mmproj_part = f" {mmproj_arg} {mmproj_path}" if mmproj_path else ""

    # Check if macro_name already contains ${...} references (for complex macros)
    if macro_name.startswith("${") and macro_name.endswith("}"):
        # It's already a formatted macro expression, use as-is
        cmd = f"${{binary}} -m {model_path} --port ${{PORT}} --host 0.0.0.0{mmproj_part} {macro_name}"
    else:
        # It's a simple macro name, wrap it with ${}
        cmd = f"${{binary}} -m {model_path} --port ${{PORT}} --host 0.0.0.0{mmproj_part} ${{{macro_name}}}"
    return MultilineLiteral(cmd)


def ensure_unique_model_name(model_name: str, model_id: str, name_to_id: dict[str, str]) -> None:
    existing_id = name_to_id.get(model_name)
    if existing_id and existing_id != model_id:
        raise ValueError(
            f"Duplicate model name '{model_name}' generated for model IDs '{existing_id}' and '{model_id}'"
        )
    name_to_id[model_name] = model_id


def extract_ngl(cmd: str) -> int:
    """Extract -ngl value from command string. Returns 0 if not found."""
    match = NGL_PATTERN.search(cmd)
    return int(match.group(1)) if match else 0


def extract_context_length(cmd: str, fallback: int) -> int:
    """Extract -c / --ctx-size value from command string. Returns fallback if not found."""
    match = CONTEXT_PATTERN.search(cmd)
    return int(match.group(1)) if match else fallback


def expand_macro_expression(expression: str, all_macros: dict[str, str]) -> str:
    """Expand either a macro name or an arbitrary expression containing ${...} references."""
    if expression in all_macros:
        return expand_macro(expression, all_macros)

    expanded = expression
    for macro_name in re.findall(r"\$\{([^}]+)\}", expression):
        if macro_name in all_macros:
            expanded = expanded.replace(f"${{{macro_name}}}", expand_macro(macro_name, all_macros))
    return deduplicate_parameters(expanded)


def cache_type_to_bytes(cache_type: str | None, default: float = 2.0) -> float:
    """Map llama.cpp cache type names to approximate bytes per element."""
    if not cache_type:
        return default

    normalized = cache_type.lower()
    mapping = {
        "f16": 2.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "q8_0": 1.0,
        "q8": 1.0,
        "q6_k": 0.75,
        "q5_0": 0.625,
        "q5_1": 0.625,
        "q5_k": 0.625,
        "q4_0": 0.5,
        "q4_1": 0.5,
        "q4_k": 0.5,
    }
    return mapping.get(normalized, default)


def extract_cache_type_bytes(cmd: str) -> tuple[float, float]:
    k_match = CACHE_TYPE_K_PATTERN.search(cmd)
    v_match = CACHE_TYPE_V_PATTERN.search(cmd)
    return (
        cache_type_to_bytes(k_match.group(1) if k_match else None),
        cache_type_to_bytes(v_match.group(1) if v_match else None),
    )


def estimate_cpu_offload_gpu_ratio(metadata, cmd: str) -> float | None:
    """Estimate the fraction of model weights that remain on GPU after CPU offload flags."""
    if not CPU_OFFLOAD_PATTERN.search(cmd):
        return 1.0

    expert_count = getattr(metadata, "expert_count", 0)
    expert_ffn = getattr(metadata, "expert_feed_forward_length", 0)
    dense_ffn = getattr(metadata, "feed_forward_length", 0)
    shared_ffn = getattr(metadata, "expert_shared_feed_forward_length", 0)

    if expert_count <= 0 or expert_ffn <= 0:
        return None

    expert_total = expert_count * expert_ffn
    total_ffn = expert_total + dense_ffn + shared_ffn
    if total_ffn <= 0:
        return None

    moe_weight_share = expert_total / total_ffn

    if re.search(r"-ot\b[^\n]*ffn_.*_exps\.\=CPU", cmd):
        offloaded_expert_fraction = 1.0
    elif "--cpu-moe" in cmd and "--n-cpu-moe" not in cmd:
        offloaded_expert_fraction = 1.0
    else:
        match = N_CPU_MOE_PATTERN.search(cmd)
        if not match:
            return None
        offloaded_expert_fraction = min(1.0, int(match.group(1)) / expert_count)

    gpu_ratio = 1.0 - (moe_weight_share * offloaded_expert_fraction)
    return max(0.0, min(1.0, gpu_ratio))


def is_vram_estimate_low_confidence(metadata, cmd: str) -> bool:
    return (
        metadata.num_layers == 0
        or metadata.num_heads == 0
        or metadata.num_heads_kv == 0
        or metadata.head_dim == 0
        or (bool(CPU_OFFLOAD_PATTERN.search(cmd)) and estimate_cpu_offload_gpu_ratio(metadata, cmd) is None)
    )


def build_vram_label(
    path_model: Path,
    cmd: str,
    metadata_fallback_ctx: int,
    cache: GGUFMetadataCache,
    mmproj_path: Path | None = None,
) -> str | None:
    """Return a VRAM label like '[12.3 GB]', or None if estimation fails."""
    try:
        metadata = get_gguf_metadata(path_model, cache)
        ngl = extract_ngl(cmd)
        ctx = extract_context_length(cmd, metadata_fallback_ctx or metadata.context_length or 4096)
        k_cache_bytes, v_cache_bytes = extract_cache_type_bytes(cmd)
        extra_gpu_bytes = mmproj_path.stat().st_size if mmproj_path else 0
        cpu_gpu_ratio = estimate_cpu_offload_gpu_ratio(metadata, cmd)
        effective_file_size = path_model.stat().st_size
        if cpu_gpu_ratio is not None:
            effective_file_size = int(effective_file_size * cpu_gpu_ratio)
        vram_gb = estimate_vram_gb(
            metadata=metadata,
            file_size_bytes=effective_file_size,
            ngl=ngl,
            context_length=ctx,
            k_cache_bytes=k_cache_bytes,
            v_cache_bytes=v_cache_bytes,
            extra_gpu_bytes=extra_gpu_bytes,
        )
        confidence_suffix = "?" if is_vram_estimate_low_confidence(metadata, cmd) else ""
        label = f"[{vram_gb:.1f} GB{confidence_suffix}]"
        logger.info(
            "VRAM estimate for %s: %s (ngl=%d, ctx=%d, cache-k=%.3f, cache-v=%.3f, mmproj=%d, cpu-gpu-ratio=%s)",
            path_model.name,
            label,
            ngl,
            ctx,
            k_cache_bytes,
            v_cache_bytes,
            extra_gpu_bytes,
            f"{cpu_gpu_ratio:.3f}" if cpu_gpu_ratio is not None else "unknown",
        )
        return label
    except Exception as e:
        logger.warning("Could not estimate VRAM for %s: %s", path_model.name, e)
        return None


def generate_model_configs(settings: Settings, config: Config) -> dict[str, YamlModelConfig]:
    # Load macro configuration
    macro_config = load_macro_config(settings.config_file)

    models = {}
    ids = set()
    name_to_id: dict[str, str] = {}
    mmproj_config = config.mmproj
    mmproj_overrides: dict[str, Path] = {}
    for key, value in mmproj_config.overrides.items():
        resolved = value if value.is_absolute() else (settings.config_file.parent / value).resolve()
        if not resolved.exists():
            raise ValueError(f"mmproj override path does not exist for '{key}': {resolved}")
        mmproj_overrides[key] = resolved

    metadata_cache = GGUFMetadataCache.load() if settings.vram_estimation else None
    cache_dirty = False

    for models_dir in settings.models_dirs:
        if not models_dir.exists():
            continue

        discovered = sorted(set(models_dir.rglob("*.gguf")) | set(models_dir.rglob("*.GGUF")))
        if mmproj_config.enabled:
            model_files = [path for path in discovered if not is_mmproj_file(path)]
            mmproj_files = [path for path in discovered if is_mmproj_file(path)]
        else:
            model_files = discovered
            mmproj_files = []

        ignore_first_segment = should_ignore_first_segment(models_dir, model_files)

        mmproj_by_prefix: dict[str, list[Path]] = {}
        for mmproj_path in mmproj_files:
            prefix = str(mmproj_path.parent)
            mmproj_by_prefix.setdefault(prefix, []).append(mmproj_path)

        for path_model in model_files:
            display_name = build_display_name(models_dir, path_model, ignore_first_segment)
            quantization = extract_quantization_suffix(path_model.name)
            model_id = f"{display_name}:{quantization}"
            model_name = build_model_name(display_name, quantization)

            if model_id in ids:
                continue
            ids.add(model_id)

            macro_name = get_model_macro(display_name, macro_config)
            selected_mmproj_path = select_mmproj_path_for_model(
                model_path=path_model,
                model_id=model_id,
                display_name=display_name,
                mmproj_overrides=mmproj_overrides,
                mmproj_by_prefix=mmproj_by_prefix,
                auto_attach=mmproj_config.auto_attach,
            )

            cmd = format_command_with_macro(
                str(path_model),
                macro_name,
                mmproj_path=str(selected_mmproj_path) if selected_mmproj_path else None,
                mmproj_arg=mmproj_config.arg,
            )

            # Expand macro to resolve -ngl and -c values for VRAM estimation
            vram_label = None
            if metadata_cache is not None:
                expanded_cmd = expand_macro_expression(macro_name, macro_config.macros)
                before_count = len(metadata_cache.entries)
                vram_label = build_vram_label(
                    path_model,
                    expanded_cmd,
                    0,
                    metadata_cache,
                    mmproj_path=selected_mmproj_path,
                )
                if len(metadata_cache.entries) != before_count:
                    cache_dirty = True
            full_name = f"{model_name} {vram_label}" if vram_label else model_name

            ensure_unique_model_name(full_name, model_id, name_to_id)
            models[model_id] = YamlModelConfig(ttl=settings.default_ttl, cmd=cmd, name=full_name)
            if selected_mmproj_path and mmproj_config.generate_no_mmproj_variant:
                no_mmproj_id = f"{model_id}-{format_suffix_for_id(mmproj_config.no_mmproj_suffix)}"
                no_mmproj_cmd = format_command_with_macro(str(path_model), macro_name)
                no_mmproj_name = f"{full_name}{mmproj_config.no_mmproj_suffix}"
                ensure_unique_model_name(no_mmproj_name, no_mmproj_id, name_to_id)
                models[no_mmproj_id] = YamlModelConfig(
                    ttl=settings.default_ttl,
                    cmd=no_mmproj_cmd,
                    name=no_mmproj_name,
                )

            # Generate variant models
            for variant in macro_config.variants:
                base_pattern = variant.get("base_pattern", "")
                suffix = variant.get("suffix", "")
                variant_macro = variant.get("macro", "")

                if base_pattern.lower() in display_name.lower() and suffix and variant_macro:
                    # Generate variant_id in YAML key suitable format from model_id
                    cleaned_suffix = format_suffix_for_id(suffix)
                    variant_id = f"{model_id}-{cleaned_suffix}"
                    variant_display_name = f"{full_name}{suffix}"
                    if variant_id not in models:  # Avoid duplicates
                        variant_cmd = format_command_with_macro(
                            str(path_model),
                            variant_macro,
                            mmproj_path=str(selected_mmproj_path) if selected_mmproj_path else None,
                            mmproj_arg=mmproj_config.arg,
                        )
                        # Estimate VRAM for variant (different macro = different ngl possibly)
                        variant_vram_label = None
                        if metadata_cache is not None:
                            expanded_variant_cmd = expand_macro_expression(variant_macro, macro_config.macros)
                            before_count = len(metadata_cache.entries)
                            variant_vram_label = build_vram_label(
                                path_model,
                                expanded_variant_cmd,
                                0,
                                metadata_cache,
                                mmproj_path=selected_mmproj_path,
                            )
                            if len(metadata_cache.entries) != before_count:
                                cache_dirty = True
                        variant_full_name = (
                            f"{model_name}{suffix} {variant_vram_label}" if variant_vram_label else variant_display_name
                        )
                        ensure_unique_model_name(variant_full_name, variant_id, name_to_id)
                        models[variant_id] = YamlModelConfig(
                            ttl=settings.default_ttl, cmd=variant_cmd, name=variant_full_name
                        )
                    if selected_mmproj_path and mmproj_config.generate_no_mmproj_variant:
                        no_mmproj_variant_id = f"{variant_id}-{format_suffix_for_id(mmproj_config.no_mmproj_suffix)}"
                        if no_mmproj_variant_id not in models:
                            no_mmproj_variant_cmd = format_command_with_macro(str(path_model), variant_macro)
                            existing_variant = models.get(variant_id)
                            base_variant_name = existing_variant.name if existing_variant else variant_display_name
                            no_mmproj_variant_name = f"{base_variant_name}{mmproj_config.no_mmproj_suffix}"
                            ensure_unique_model_name(no_mmproj_variant_name, no_mmproj_variant_id, name_to_id)
                            models[no_mmproj_variant_id] = YamlModelConfig(
                                ttl=settings.default_ttl,
                                cmd=no_mmproj_variant_cmd,
                                name=no_mmproj_variant_name,
                            )

    if cache_dirty and metadata_cache is not None:
        metadata_cache.save()

    if not models:
        raise ValueError("No models found. Please check your models directory and ensure .gguf files exist.")

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
    models = generate_model_configs(settings, config)
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
