# `base.yaml` Specification

This document defines the input configuration format (`base.yaml`) used by `llama-swap-config-autogen`.

## 1. Purpose

`base.yaml` is the source-of-truth file for generating a full `llama-swap` runtime config (`config.yaml`).

Generation flow:

1. Scan model directories from `models`.
2. Discover `*.gguf` files.
3. Build model IDs and model names from the relative directory path and filename quantization suffix.
4. Select a macro via `model_patterns` (or fallback to `default-params`).
5. Apply optional `variants`.
6. Emit only macros actually referenced by generated model commands.

## 2. Top-Level Schema

```yaml
models: [<path>, ...]                # required
macros: { <name>: <string>, ... }    # optional
model_patterns: { <substring>: <macro>, ... }  # optional
variants:                            # optional
  - base_pattern: <substring>
    suffix: <display suffix>
    macro: <macro name or macro expression>
mmproj:                              # optional
  enabled: <bool>                    # default: true
  auto_attach: <bool>                # default: true
  arg: <string>                      # default: --mmproj
  generate_no_mmproj_variant: <bool> # default: true
  no_mmproj_suffix: <string>         # default: " (no mmproj)"
  overrides: { <model-id|display-name|filename>: <path>, ... }

default_ttl: <int>                   # optional, default: 300
health_check_timeout: <int>          # optional, default: 240
log_level: <string>                  # optional, default: info
start_port: <int>                    # optional, default: 9091
```

## 3. Field Definitions

### 3.1 `models` (required)

- Type: `list[path]`
- Meaning: Directories to scan for `.gguf` files.
- Behavior: Non-existent directories are skipped.
- Directory layout requirements:
    - `<models_dir>/<model>/<file>.gguf`
    - `<models_dir>/<model>/<variant>/<file>.gguf`
    - `<models_dir>/<family>/<model>/<file>.gguf`
    - `<models_dir>/<family>/<model>/<variant>/<file>.gguf`
- When the family layout is used, the first path segment is ignored for naming.
- A single `models` entry must use one layout style consistently.
- Any `.gguf` file outside those depths, or any mixed style under one root, causes generation to fail with an error.

### 3.2 `macros` (optional)

- Type: `map[string, string]`
- Meaning: Reusable command fragments.
- Macro references use `${name}` syntax.
- Macro names are normalized for output compatibility:
    - `.` is converted to `-`
    - references are also normalized

### 3.3 `model_patterns` (optional)

- Type: `map[string, string]`
- Key: Substring pattern matched against generated display name.
- Value: Macro name, or macro expression (e.g. `${a} ${b}`).
- First matching entry is selected (in file order).
- Fallback if no match: `default-params`.

### 3.4 `variants` (optional)

- Type: `list[object]`
- Required keys per item:
    - `base_pattern` (substring matched against display name, case-insensitive)
    - `suffix` (display name suffix)
    - `macro` (macro name or macro expression)
- Behavior: Generates extra model entries for matching models.

### 3.5 Runtime defaults

- `default_ttl`: `300`
- `health_check_timeout`: `240`
- `log_level`: `"info"`
- `start_port`: `9091`

### 3.6 `mmproj` (optional)

- Type: `object`
- Fields:
    - `enabled` (`bool`, default `true`)
    - `auto_attach` (`bool`, default `true`)
    - `arg` (`string`, default `--mmproj`)
    - `generate_no_mmproj_variant` (`bool`, default `true`)
    - `no_mmproj_suffix` (`string`, default `" (no mmproj)"`)
    - `overrides` (`map[string, path]`, default `{}`)

Behavior when `enabled: true`:

- Files whose names contain `mmproj` are excluded from standalone model generation.
- `--mmproj <path>` is appended to model commands when:
    - an override exists for model ID, display name, or model filename, or
    - `auto_attach: true` and there is exactly one `mmproj` candidate in the same directory as the model file.
- If `generate_no_mmproj_variant: true`, an additional model entry without `--mmproj` is generated for each model (and variant)
  that had `mmproj` attached.

Behavior when `enabled: false`:

- Legacy behavior is kept: `mmproj` files are treated like normal `.gguf` model files.

## 4. Macro Resolution Rules

## 4.1 Reference syntax

- References must use `${macro-name}`.
- Nested references are expanded recursively.

## 4.2 Expansion behavior

- Circular references are detected.
- Duplicate CLI flags are deduplicated with rightmost priority.
    - Example: if both `--ctx-size 32768` and `--ctx-size 65536` appear, the latter wins.

## 4.3 Command composition

Generated model command format:

```text
${binary} -m <absolute_model_path> --port ${PORT} --host 0.0.0.0 <macro>
```

If `<macro>` is a macro name, it is wrapped as `${<macro>}`.
If `<macro>` is already an expression like `${a} ${b}`, it is used as-is.

## 5. Model Discovery and Naming

For each discovered `.gguf` file:

- Display name: derived from the relative directory path and lowercased.
- Model ID: derived as `<relative-directory-path-lowercased>:<format>`.
- Model name: derived as `<relative-directory-path-lowercased>:<format>`.
- Quantization format: extracted from the filename suffix such as `Q4_K_M`, `Q8_0`, `BF16`, or `F16`.
- Generated display names must be unique across all emitted model entries.
- Duplicate model IDs are ignored after first occurrence.

Examples:

- `Qwen3-30B/Instruct-2507/model-Q4_K_M.gguf` -> model name `qwen3-30b/instruct-2507:Q4_K_M`
- `Qwen3-30B/Instruct-2507/model-Q4_K_M.gguf` -> model ID `qwen3-30b/instruct-2507:Q4_K_M`
- `Gemma-4-31B/it/model-Q4_K_M.gguf` -> model name `gemma-4-31b/it:Q4_K_M`

Note: Filenames no longer need to follow a `user_repo_...` naming convention, but they must still contain a recognizable
quantization suffix for model ID generation.

## 6. Generated Output Mapping

`base.yaml` fields map to generated `config.yaml` as:

- `health_check_timeout` -> `healthCheckTimeout`
- `log_level` -> `logLevel`
- `start_port` -> `startPort`
- `default_ttl` -> each model entry `ttl`
- `macros` -> filtered to only those referenced by generated model commands

## 7. Recommended Authoring Rules

1. Define `binary` and `default-params` in `macros`.
2. Keep macro names stable and kebab-case-like (`a-z`, `0-9`, `_`, `-`).
3. Build macros hierarchically (`parts` -> `presets`).
4. Keep `model_patterns` for default assignment only.
5. Put optional behavior differences in `variants`.
6. Keep pattern keys specific enough to avoid accidental matches.

## 8. Minimal Example

```yaml
models:
  - /opt/data/llm/models

default_ttl: 300
health_check_timeout: 240
log_level: warn
start_port: 9091

macros:
  binary: /app/llama-server
  common-base: --jinja --flash-attn on
  layers-default: --n-gpu-layers 999
  context-default: --ctx-size 32768
  default-params: ${common-base} ${layers-default} ${context-default}
  qwen-cpu: ${default-params} --n-cpu-moe 12 --threads 16 --ctx-size 65536

model_patterns:
  qwen3: default-params

variants:
  - base_pattern: qwen3
    suffix: " (with CPU)"
    macro: qwen-cpu
```

## 9. Compatibility Notes

- This spec describes the current behavior of the codebase in this repository.
- Unknown top-level keys are ignored by generation logic.
- If no models are discovered, generation fails.
