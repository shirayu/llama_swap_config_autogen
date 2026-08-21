"""YAML configuration generation logic."""

import fnmatch
import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import load_macro_config
from .fit_params import FitParamsCache, apply_path_prefix_map, estimate_vram_gib_via_fit_params
from .gguf_metadata import GGUFMetadataCache, get_gguf_metadata, read_mmproj_modalities
from .models import (
    CapabilitiesConfig,
    Config,
    MacroConfig,
    ModelPatternConfig,
    MultilineLiteral,
    Settings,
    YamlModelConfig,
)

logger = logging.getLogger(__name__)

MMPROJ_PATTERN = re.compile(r"mmproj", re.IGNORECASE)
BF16_PATTERN = re.compile(r"bf16", re.IGNORECASE)
F16_PATTERN = re.compile(r"(?<!b)f16", re.IGNORECASE)
F32_PATTERN = re.compile(r"f32", re.IGNORECASE)
QUANTIZATION_PATTERN = re.compile(r"-(Q\d(?:_[A-Z0-9]+)+|BF16|F16)(?=\.gguf$)", re.IGNORECASE)
NGL_PATTERN = re.compile(r"(?:-ngl|--n-gpu-layers)\s+(\d+)")
CONTEXT_PATTERN = re.compile(r"(?:-c|--ctx-size)\s+(\d+)")
CACHE_TYPE_K_PATTERN = re.compile(r"--cache-type-k\s+([^\s]+)")
CACHE_TYPE_V_PATTERN = re.compile(r"--cache-type-v\s+([^\s]+)")
CPU_OFFLOAD_PATTERN = re.compile(r"(?:--cpu-moe|--n-cpu-moe\b|-ot\b[^\n]*=CPU)", re.IGNORECASE)


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


def matches_model_pattern(pattern: str | list[str], *model_identifiers: str) -> bool:
    if isinstance(pattern, list):
        return any(matches_model_pattern(p, *model_identifiers) for p in pattern)
    normalized_pattern = pattern.lower()
    if any(ch in normalized_pattern for ch in "*?["):
        return any(fnmatch.fnmatchcase(identifier.lower(), normalized_pattern) for identifier in model_identifiers)
    return any(normalized_pattern in identifier.lower() for identifier in model_identifiers)


def get_model_pattern_config(model_name: str, macro_config: MacroConfig, *model_identifiers: str) -> ModelPatternConfig:
    """Get appropriate model pattern configuration based on model identifiers."""
    match_targets = (*model_identifiers, model_name)
    for pattern, pattern_config in macro_config.model_patterns.items():
        if matches_model_pattern(pattern, *match_targets):
            return pattern_config

    return ModelPatternConfig(macro="default-params")


def get_model_macro(model_name: str, macro_config: MacroConfig, *model_identifiers: str) -> str:
    """Get appropriate macro based on model identifiers."""
    return get_model_pattern_config(model_name, macro_config, *model_identifiers).macro


def is_mmproj_file(path_model: Path) -> bool:
    return bool(MMPROJ_PATTERN.search(path_model.name))


def resolve_mmproj_path(value_str: str, config_dir: Path, all_mmproj_files: list[Path]) -> Path:
    value = Path(value_str)
    if value.is_absolute():
        resolved = value
    else:
        resolved = (config_dir / value).resolve()

    if resolved.exists():
        return resolved

    candidates = []
    norm_val = value_str.replace("\\", "/").lower()
    for mmproj_path in all_mmproj_files:
        norm_path = mmproj_path.as_posix().lower()
        if mmproj_path.name.lower() == norm_val or norm_path.endswith("/" + norm_val) or norm_path.endswith(norm_val):
            candidates.append(mmproj_path)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        raise ValueError(
            f"mmproj path is ambiguous: '{value_str}'. Matches multiple files: {[str(c) for c in candidates]}"
        )
    else:
        raise ValueError(f"mmproj path or file name does not exist: {value_str}")


def select_mmproj_path_for_model(
    model_path: Path,
    model_id: str,
    display_name: str,
    mmproj_overrides: dict[str, Path],
    mmproj_by_prefix: dict[str, list[Path]],
    auto_attach: bool,
    pattern_mmproj_path: Path | None = None,
) -> Path | None:
    if pattern_mmproj_path:
        return pattern_mmproj_path

    exact_override = (
        mmproj_overrides.get(model_id) or mmproj_overrides.get(display_name) or mmproj_overrides.get(model_path.name)
    )
    if exact_override:
        return exact_override

    if not auto_attach:
        return None

    candidates = mmproj_by_prefix.get(str(model_path.parent), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for precision_pattern in (BF16_PATTERN, F16_PATTERN, F32_PATTERN):
            precision_candidates = [candidate for candidate in candidates if precision_pattern.search(candidate.name)]
            if len(precision_candidates) == 1:
                return precision_candidates[0]
            if len(precision_candidates) > 1:
                return None

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
    if not match:
        return fallback
    value = int(match.group(1))
    return fallback if value == 0 else value


def expand_macro_expression(expression: str, all_macros: dict[str, str]) -> str:
    """Expand either a macro name or an arbitrary expression containing ${...} references."""
    expression_name = expression.split(":", 1)[0] if ":" in expression else expression
    if expression_name in all_macros or expression in all_macros:
        return expand_macro(expression, all_macros)

    expanded = expression
    for macro_match in re.findall(r"\$\{([^}]+)\}", expression):
        macro_name = macro_match.split(":", 1)[0] if ":" in macro_match else macro_match
        if macro_name in all_macros:
            expanded = expanded.replace(f"${{{macro_match}}}", expand_macro(macro_match, all_macros))
    return deduplicate_parameters(expanded)


def extract_extra_fit_args(cmd: str) -> list[str]:
    """Extract CPU-offload-related flags (-ot, --cpu-moe, --n-cpu-moe) to forward to fit-params.

    fit-params simulates the actual model load, so passing these through directly lets it compute
    the resulting GPU/CPU tensor split instead of us approximating it.
    """
    extra_args = []
    for match in re.finditer(r"-ot\s+(\S+)", cmd):
        extra_args += ["-ot", match.group(1)]
    if re.search(r"--cpu-moe\b", cmd):
        extra_args.append("--cpu-moe")
    n_cpu_moe_match = re.search(r"--n-cpu-moe\s+(\d+)", cmd)
    if n_cpu_moe_match:
        extra_args += ["--n-cpu-moe", n_cpu_moe_match.group(1)]
    return extra_args


def estimate_vram_gib(
    path_model: Path,
    cmd: str,
    metadata_fallback_ctx: int,
    metadata_cache: GGUFMetadataCache,
    fit_params_cache: FitParamsCache,
    llama_bin: list[str] | None,
    path_prefix_map: dict[str, str],
    mmproj_path: Path | None = None,
) -> float | None:
    """Return the GPU VRAM estimate in GiB via llama.cpp fit-params, or None if estimation fails/unavailable."""
    if not llama_bin:
        return None
    try:
        metadata = get_gguf_metadata(path_model, metadata_cache)
        ngl = extract_ngl(cmd)
        ctx = extract_context_length(cmd, metadata_fallback_ctx or metadata.context_length or 4096)
        k_match = CACHE_TYPE_K_PATTERN.search(cmd)
        v_match = CACHE_TYPE_V_PATTERN.search(cmd)
        extra_args = extract_extra_fit_args(cmd)
        extra_gpu_gib = (mmproj_path.stat().st_size / 1024**3) if mmproj_path else 0.0

        vram_gib = estimate_vram_gib_via_fit_params(
            llama_bin=llama_bin,
            path_model=path_model,
            ngl=ngl,
            ctx=ctx,
            cache_type_k=k_match.group(1) if k_match else None,
            cache_type_v=v_match.group(1) if v_match else None,
            extra_args=extra_args,
            path_prefix_map=path_prefix_map,
            cache=fit_params_cache,
            extra_gpu_gib=extra_gpu_gib,
        )
        if vram_gib is not None:
            logger.info(
                "VRAM estimate for %s: %.1f GiB (ngl=%d, ctx=%d, mmproj=%.2f GiB)",
                path_model.name,
                vram_gib,
                ngl,
                ctx,
                extra_gpu_gib,
            )
        return vram_gib
    except Exception as e:
        logger.warning("Could not estimate VRAM for %s: %s", path_model.name, e)
        return None


def build_model_metadata(
    display_name: str,
    path_model: Path,
    expanded_cmd: str,
    metadata_cache: GGUFMetadataCache | None,
    fit_params_cache: FitParamsCache | None = None,
    llama_bin: list[str] | None = None,
    path_prefix_map: dict[str, str] | None = None,
    mmproj_path: Path | None = None,
    vram_estimation: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Build model metadata and report whether the GGUF cache changed.

    llama-swap wraps this config-level metadata under ``meta.llamaswap`` in
    its /v1/models response, so this function must return the inner mapping.
    """
    model_metadata: dict[str, Any] = {
        "model_family": display_name.split("/", 1)[0],
    }
    cache_changed = False
    if metadata_cache is not None:
        before_count = len(metadata_cache.entries)
        if vram_estimation and fit_params_cache is not None:
            vram_gib = estimate_vram_gib(
                path_model,
                expanded_cmd,
                0,
                metadata_cache,
                fit_params_cache,
                llama_bin,
                path_prefix_map or {},
                mmproj_path=mmproj_path,
            )
            if vram_gib is not None:
                model_metadata["estimated_vram_bytes"] = round(vram_gib * 1024**3)
        reasoning_supported = resolve_reasoning_support(path_model, metadata_cache)
        if reasoning_supported is not None:
            model_metadata["reasoning_supported"] = reasoning_supported
        try:
            metadata = get_gguf_metadata(path_model, metadata_cache)
            model_metadata["file_size_bytes"] = path_model.stat().st_size
            if metadata.expert_count > 0:
                model_metadata["expert_count"] = metadata.expert_count
                model_metadata["expert_used_count"] = metadata.expert_used_count
            if metadata.repo_url:
                model_metadata["repo_url"] = metadata.repo_url
            if metadata.license:
                model_metadata["license"] = metadata.license
        except Exception as e:
            logger.warning("Could not read GGUF metadata for %s: %s", path_model.name, e)
        cache_changed = len(metadata_cache.entries) != before_count
    if mmproj_path is not None:
        projector_type = read_mmproj_modalities(mmproj_path).projector_type
        if projector_type:
            model_metadata["mmproj_projector_type"] = projector_type
    model_metadata.update(load_sidecar_metadata(path_model))
    return model_metadata, cache_changed


def load_sidecar_metadata(path_model: Path) -> dict[str, Any]:
    """Load a user-authored `<model>.json` next to the GGUF and merge it into metadata.

    The sidecar is optional and its contents fully override auto-derived fields on conflict.
    """
    sidecar_path = path_model.with_suffix(".json")
    if not sidecar_path.is_file():
        return {}
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read sidecar metadata %s: %s", sidecar_path, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("Sidecar metadata %s is not a JSON object, ignoring", sidecar_path)
        return {}
    return data


def resolve_context_length(
    macro_expr: str,
    macro_config: MacroConfig,
    path_model: Path,
    metadata_cache: GGUFMetadataCache | None,
) -> int | None:
    """Determine context length from an expanded macro's -c/--ctx-size, falling back to GGUF metadata."""
    expanded_cmd = expand_macro_expression(macro_expr, macro_config.macros)
    match = CONTEXT_PATTERN.search(expanded_cmd)
    if match:
        ctx = int(match.group(1))
        if ctx > 0:
            return ctx

    if metadata_cache is None:
        return None

    try:
        metadata = get_gguf_metadata(path_model, metadata_cache)
    except Exception as e:
        logger.warning("Could not read GGUF metadata for %s: %s", path_model.name, e)
        return None

    return metadata.context_length if metadata.context_length > 0 else None


def resolve_tool_support(
    path_model: Path,
    metadata_cache: GGUFMetadataCache | None,
) -> bool | None:
    """Detect tool-calling support from the GGUF chat_template.

    Requires vram_estimation or read_gguf_metadata to read GGUF metadata.
    """
    if metadata_cache is None:
        return None

    try:
        metadata = get_gguf_metadata(path_model, metadata_cache)
    except Exception as e:
        logger.warning("Could not read GGUF metadata for %s: %s", path_model.name, e)
        return None

    return metadata.supports_tools or None


def resolve_reasoning_support(
    path_model: Path,
    metadata_cache: GGUFMetadataCache | None,
) -> bool | None:
    """Detect reasoning/thinking support from the GGUF chat_template.

    Requires vram_estimation or read_gguf_metadata to read GGUF metadata.
    """
    if metadata_cache is None:
        return None

    try:
        metadata = get_gguf_metadata(path_model, metadata_cache)
    except Exception as e:
        logger.warning("Could not read GGUF metadata for %s: %s", path_model.name, e)
        return None

    return metadata.supports_reasoning


def build_capabilities(
    macro_expr: str,
    macro_config: MacroConfig,
    path_model: Path,
    metadata_cache: GGUFMetadataCache | None,
    user_capabilities: CapabilitiesConfig | None,
    mmproj_path: Path | None = None,
) -> CapabilitiesConfig | None:
    """Build capabilities for a model entry: auto-derived fields merged with user-specified fields.

    Auto-derived: context (from -c/--ctx-size or GGUF metadata), tools (from GGUF chat_template),
    in (text, plus image/audio depending on the attached mmproj's clip.has_vision_encoder /
    clip.has_audio_encoder metadata).
    """
    context = resolve_context_length(macro_expr, macro_config, path_model, metadata_cache)
    tools = resolve_tool_support(path_model, metadata_cache)
    auto_in = ["text"]
    if mmproj_path is not None:
        modalities = read_mmproj_modalities(mmproj_path)
        if modalities.has_vision:
            auto_in.append("image")
        if modalities.has_audio:
            auto_in.append("audio")
        if not modalities.has_vision and not modalities.has_audio:
            auto_in.append("image")

    merged = user_capabilities.model_copy() if user_capabilities is not None else CapabilitiesConfig()
    if merged.context is None and context is not None:
        merged.context = context
    if merged.tools is None and tools is not None:
        merged.tools = tools
    if merged.in_ is None:
        merged.in_ = auto_in

    return merged if merged.model_dump(exclude_none=True) else None


def resolve_variant_macro_template(template: str, pattern_config: ModelPatternConfig) -> str:
    """Resolve placeholders like ${cpu-macro} in a variant macro template using pattern_config's extra fields."""
    import re

    def replace_placeholder(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if pattern_config.model_extra and var_name in pattern_config.model_extra:
            return str(pattern_config.model_extra[var_name])
        return f"${{{var_name}}}"

    return re.sub(r"\$\{([^}]+)\}", replace_placeholder, template)


def generate_model_configs(settings: Settings, config: Config) -> dict[str, YamlModelConfig]:
    # Load macro configuration
    macro_config = load_macro_config(settings.config_file)

    models = {}
    ids = set()
    name_to_id: dict[str, str] = {}
    mmproj_config = config.mmproj

    # Pre-scan all mmproj files across models directories for potential override resolution
    all_mmproj_files: list[Path] = []
    if mmproj_config.enabled:
        for models_dir in settings.models_dirs:
            if models_dir.exists():
                discovered_files = set(models_dir.rglob("*.gguf")) | set(models_dir.rglob("*.GGUF"))
                all_mmproj_files.extend([path for path in discovered_files if is_mmproj_file(path)])

    mmproj_overrides: dict[str, Path] = {}
    for key, value in mmproj_config.overrides.items():
        try:
            mmproj_overrides[key] = resolve_mmproj_path(str(value), settings.config_file.parent, all_mmproj_files)
        except ValueError as exc:
            raise ValueError(f"mmproj override error for '{key}': {exc}") from exc

    metadata_cache = GGUFMetadataCache.load() if settings.read_gguf_metadata else None
    fit_params_cache = FitParamsCache.load() if settings.vram_estimation and settings.llama_bin else None
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

            pattern_config = get_model_pattern_config(display_name, macro_config, model_id, path_model.name)
            macro_name = pattern_config.macro

            pattern_mmproj_path = None
            pattern_mmproj_val = getattr(pattern_config, "mmproj", None)
            if pattern_mmproj_val is not None:
                try:
                    pattern_mmproj_path = resolve_mmproj_path(
                        pattern_mmproj_val, settings.config_file.parent, all_mmproj_files
                    )
                except ValueError as exc:
                    raise ValueError(f"mmproj resolution error in model pattern for '{display_name}': {exc}") from exc

            runtime_model_path = apply_path_prefix_map(path_model, settings.path_prefix_map)

            selected_mmproj_path = select_mmproj_path_for_model(
                model_path=path_model,
                model_id=model_id,
                display_name=display_name,
                mmproj_overrides=mmproj_overrides,
                mmproj_by_prefix=mmproj_by_prefix,
                auto_attach=mmproj_config.auto_attach,
                pattern_mmproj_path=pattern_mmproj_path,
            )
            runtime_mmproj_path = (
                apply_path_prefix_map(selected_mmproj_path, settings.path_prefix_map) if selected_mmproj_path else None
            )
            if pattern_config.emit_base:
                cmd = format_command_with_macro(
                    runtime_model_path,
                    macro_name,
                    mmproj_path=runtime_mmproj_path,
                    mmproj_arg=mmproj_config.arg,
                )

                expanded_cmd = expand_macro_expression(macro_name, macro_config.macros)
                base_metadata, metadata_changed = build_model_metadata(
                    display_name,
                    path_model,
                    expanded_cmd,
                    metadata_cache,
                    mmproj_path=selected_mmproj_path,
                    fit_params_cache=fit_params_cache,
                    llama_bin=settings.llama_bin,
                    path_prefix_map=settings.path_prefix_map,
                    vram_estimation=settings.vram_estimation,
                )
                if metadata_changed:
                    cache_dirty = True
                full_name = model_name
                base_capabilities = build_capabilities(
                    macro_name,
                    macro_config,
                    path_model,
                    metadata_cache,
                    pattern_config.capabilities,
                    mmproj_path=selected_mmproj_path,
                )

                ensure_unique_model_name(full_name, model_id, name_to_id)
                models[model_id] = YamlModelConfig(
                    ttl=settings.default_ttl,
                    cmd=cmd,
                    name=full_name,
                    capabilities=base_capabilities,
                    metadata=base_metadata,
                )
                if selected_mmproj_path and mmproj_config.generate_no_mmproj_variant:
                    no_mmproj_id = f"{model_id}-{format_suffix_for_id(mmproj_config.no_mmproj_suffix)}"
                    no_mmproj_cmd = format_command_with_macro(runtime_model_path, macro_name)
                    no_mmproj_metadata, metadata_changed = build_model_metadata(
                        display_name,
                        path_model,
                        expanded_cmd,
                        metadata_cache,
                        fit_params_cache=fit_params_cache,
                        llama_bin=settings.llama_bin,
                        path_prefix_map=settings.path_prefix_map,
                        vram_estimation=settings.vram_estimation,
                    )
                    if metadata_changed:
                        cache_dirty = True
                    no_mmproj_base_name = model_name
                    no_mmproj_name = f"{no_mmproj_base_name}{mmproj_config.no_mmproj_suffix}"
                    no_mmproj_capabilities = build_capabilities(
                        macro_name,
                        macro_config,
                        path_model,
                        metadata_cache,
                        pattern_config.capabilities,
                    )
                    ensure_unique_model_name(no_mmproj_name, no_mmproj_id, name_to_id)
                    models[no_mmproj_id] = YamlModelConfig(
                        ttl=settings.default_ttl,
                        cmd=no_mmproj_cmd,
                        name=no_mmproj_name,
                        capabilities=no_mmproj_capabilities,
                        metadata=no_mmproj_metadata,
                    )

            # Generate preset-based variant models
            if pattern_config.variants:
                for preset_name in pattern_config.variants:
                    preset_items = macro_config.variant_presets.get(preset_name, [])
                    for preset_item in preset_items:
                        suffix = preset_item.suffix
                        macro_template = preset_item.macro

                        # Resolve variables (arguments) passed from pattern_config.model_extra
                        variant_macro = resolve_variant_macro_template(macro_template, pattern_config)

                        cleaned_suffix = format_suffix_for_id(suffix)
                        variant_id = f"{model_id}-{cleaned_suffix}"
                        variant_display_name = f"{model_name}{suffix}"
                        expanded_variant_cmd = expand_macro_expression(variant_macro, macro_config.macros)

                        if variant_id not in models:
                            variant_cmd = format_command_with_macro(
                                runtime_model_path,
                                variant_macro,
                                mmproj_path=runtime_mmproj_path,
                                mmproj_arg=mmproj_config.arg,
                            )
                            variant_metadata, metadata_changed = build_model_metadata(
                                display_name,
                                path_model,
                                expanded_variant_cmd,
                                metadata_cache,
                                mmproj_path=selected_mmproj_path,
                                fit_params_cache=fit_params_cache,
                                llama_bin=settings.llama_bin,
                                path_prefix_map=settings.path_prefix_map,
                                vram_estimation=settings.vram_estimation,
                            )
                            if metadata_changed:
                                cache_dirty = True
                            variant_full_name = variant_display_name
                            variant_capabilities = build_capabilities(
                                variant_macro,
                                macro_config,
                                path_model,
                                metadata_cache,
                                pattern_config.capabilities,
                                mmproj_path=selected_mmproj_path,
                            )
                            ensure_unique_model_name(variant_full_name, variant_id, name_to_id)
                            models[variant_id] = YamlModelConfig(
                                ttl=settings.default_ttl,
                                cmd=variant_cmd,
                                name=variant_full_name,
                                capabilities=variant_capabilities,
                                metadata=variant_metadata,
                            )
                        if selected_mmproj_path and mmproj_config.generate_no_mmproj_variant:
                            no_mmproj_variant_id = (
                                f"{variant_id}-{format_suffix_for_id(mmproj_config.no_mmproj_suffix)}"
                            )
                            if no_mmproj_variant_id not in models:
                                no_mmproj_variant_cmd = format_command_with_macro(runtime_model_path, variant_macro)
                                no_mmproj_variant_metadata, metadata_changed = build_model_metadata(
                                    display_name,
                                    path_model,
                                    expanded_variant_cmd,
                                    metadata_cache,
                                    fit_params_cache=fit_params_cache,
                                    llama_bin=settings.llama_bin,
                                    path_prefix_map=settings.path_prefix_map,
                                    vram_estimation=settings.vram_estimation,
                                )
                                if metadata_changed:
                                    cache_dirty = True
                                base_variant_name = variant_display_name
                                no_mmproj_variant_name = f"{base_variant_name}{mmproj_config.no_mmproj_suffix}"
                                no_mmproj_variant_capabilities = build_capabilities(
                                    variant_macro,
                                    macro_config,
                                    path_model,
                                    metadata_cache,
                                    pattern_config.capabilities,
                                )
                                ensure_unique_model_name(no_mmproj_variant_name, no_mmproj_variant_id, name_to_id)
                                models[no_mmproj_variant_id] = YamlModelConfig(
                                    ttl=settings.default_ttl,
                                    cmd=no_mmproj_variant_cmd,
                                    name=no_mmproj_variant_name,
                                    capabilities=no_mmproj_variant_capabilities,
                                    metadata=no_mmproj_variant_metadata,
                                )

            # Generate variant models
            for variant in macro_config.variants:
                base_pattern = variant.get("base_pattern", "")
                suffix = variant.get("suffix", "")
                variant_macro = variant.get("macro", "")

                if (
                    matches_model_pattern(base_pattern, model_id, path_model.name, display_name)
                    and suffix
                    and variant_macro
                ):
                    # Generate variant_id in YAML key suitable format from model_id
                    cleaned_suffix = format_suffix_for_id(suffix)
                    variant_id = f"{model_id}-{cleaned_suffix}"
                    variant_display_name = f"{model_name}{suffix}"
                    expanded_variant_cmd = expand_macro_expression(variant_macro, macro_config.macros)
                    if variant_id not in models:  # Avoid duplicates
                        variant_cmd = format_command_with_macro(
                            runtime_model_path,
                            variant_macro,
                            mmproj_path=runtime_mmproj_path,
                            mmproj_arg=mmproj_config.arg,
                        )
                        variant_metadata, metadata_changed = build_model_metadata(
                            display_name,
                            path_model,
                            expanded_variant_cmd,
                            metadata_cache,
                            mmproj_path=selected_mmproj_path,
                            fit_params_cache=fit_params_cache,
                            llama_bin=settings.llama_bin,
                            path_prefix_map=settings.path_prefix_map,
                            vram_estimation=settings.vram_estimation,
                        )
                        if metadata_changed:
                            cache_dirty = True
                        variant_full_name = variant_display_name
                        variant_capabilities = build_capabilities(
                            variant_macro,
                            macro_config,
                            path_model,
                            metadata_cache,
                            pattern_config.capabilities,
                            mmproj_path=selected_mmproj_path,
                        )
                        ensure_unique_model_name(variant_full_name, variant_id, name_to_id)
                        models[variant_id] = YamlModelConfig(
                            ttl=settings.default_ttl,
                            cmd=variant_cmd,
                            name=variant_full_name,
                            capabilities=variant_capabilities,
                            metadata=variant_metadata,
                        )
                    if selected_mmproj_path and mmproj_config.generate_no_mmproj_variant:
                        no_mmproj_variant_id = f"{variant_id}-{format_suffix_for_id(mmproj_config.no_mmproj_suffix)}"
                        if no_mmproj_variant_id not in models:
                            no_mmproj_variant_cmd = format_command_with_macro(runtime_model_path, variant_macro)
                            no_mmproj_variant_metadata, metadata_changed = build_model_metadata(
                                display_name,
                                path_model,
                                expanded_variant_cmd,
                                metadata_cache,
                                fit_params_cache=fit_params_cache,
                                llama_bin=settings.llama_bin,
                                path_prefix_map=settings.path_prefix_map,
                                vram_estimation=settings.vram_estimation,
                            )
                            if metadata_changed:
                                cache_dirty = True
                            base_variant_name = variant_display_name
                            no_mmproj_variant_name = f"{base_variant_name}{mmproj_config.no_mmproj_suffix}"
                            no_mmproj_variant_capabilities = build_capabilities(
                                variant_macro,
                                macro_config,
                                path_model,
                                metadata_cache,
                                pattern_config.capabilities,
                            )
                            ensure_unique_model_name(no_mmproj_variant_name, no_mmproj_variant_id, name_to_id)
                            models[no_mmproj_variant_id] = YamlModelConfig(
                                ttl=settings.default_ttl,
                                cmd=no_mmproj_variant_cmd,
                                name=no_mmproj_variant_name,
                                capabilities=no_mmproj_variant_capabilities,
                                metadata=no_mmproj_variant_metadata,
                            )

    if cache_dirty and metadata_cache is not None:
        metadata_cache.save()
    if fit_params_cache is not None:
        fit_params_cache.save()

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


def expand_macro(macro_name_with_args: str, all_macros: dict[str, str], visited: set[str] | None = None) -> str:
    """Recursively expand a macro by resolving all nested macro references and positional parameters"""
    if visited is None:
        visited = set()

    # Parse macro name and arguments
    macro_name = macro_name_with_args
    args = []
    if ":" in macro_name_with_args:
        macro_name, args_str = macro_name_with_args.split(":", 1)
        args = args_str.split(",")

    if macro_name in visited:
        raise ValueError(f"Circular macro reference detected: {macro_name}")

    if macro_name not in all_macros:
        # Return as-is if macro not found (could be built-in like ${PORT})
        return f"${{{macro_name_with_args}}}"

    visited.add(macro_name)
    macro_value = all_macros[macro_name]

    # Replace positional parameters ${1}, ${2}, etc. with arguments
    for idx, arg in enumerate(args):
        macro_value = macro_value.replace(f"${{{idx + 1}}}", arg)

    # Find all nested macro references
    nested_macros = re.findall(r"\$\{([^}]+)\}", macro_value)

    # Expand each nested macro
    expanded_value = macro_value
    for nested_macro in nested_macros:
        nested_name = nested_macro.split(":", 1)[0] if ":" in nested_macro else nested_macro
        if nested_name in all_macros:
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
        base_name = macro_name.split(":", 1)[0] if ":" in macro_name else macro_name
        if base_name not in all_macros:
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
    output_config = {}

    # Merge any extra global configuration items
    # (like captureBuffer, healthCheckTimeout, logLevel, startPort) from config.model_extra
    if config.model_extra:
        for k, v in config.model_extra.items():
            if k not in {"vram_estimation", "read_gguf_metadata", "model_labels", "path_prefix_map"}:
                output_config[k] = v

    # Add model configurations and collect commands simultaneously
    output_config["models"] = {}
    all_commands = []
    for model_id, model_config in models.items():
        output_config["models"][model_id] = {
            "ttl": model_config.ttl,
            "cmd": model_config.cmd,
            "name": model_config.name,
        }
        if model_config.metadata is not None:
            output_config["models"][model_id]["metadata"] = model_config.metadata
        if model_config.capabilities is not None:
            output_config["models"][model_id]["capabilities"] = model_config.capabilities.to_yaml_dict()
        # Collect commands as strings
        cmd_str = str(model_config.cmd)
        all_commands.append(cmd_str)

    # Extract and add only used macros
    if macro_config.macros and all_commands:
        used_macros = extract_used_macros_from_commands(all_commands, macro_config.macros)
        if used_macros:
            output_config["macros"] = used_macros

    return output_config
