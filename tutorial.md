# Llama-Swap Config Autogen - Configuration Tutorial

This tutorial guides you through the advanced configuration features of `llama-swap-config-autogen`. You will learn how to design a highly maintainable, DRY (Don't Repeat Yourself) `base.yaml` that can auto-discover dozens of GGUF models and generate a complete configuration for [llama-swap](https://github.com/mostlygeek/llama-swap).

---

## Table of Contents

1. [The Basic Concept: Auto-Discovery](#1-the-basic-concept-auto-discovery)
2. [Parameterized Macros: Eliminating Numeric Duplications](#2-parameterized-macros-eliminating-numeric-duplications)
3. [Variant Presets with Implicit Argument Binding](#3-variant-presets-with-implicit-argument-binding)
4. [Model Labels with Multi-Pattern Lists](#4-model-labels-with-multi-pattern-lists)
5. [Explicit Multi-Modal Projection (mmproj) Binding](#5-explicit-multi-modal-projection-mmproj-binding)
6. [Put It All Together: A Complete base.yaml Example](#6-put-it-all-together-a-complete-baseyaml-example)

---

## 1. The Basic Concept: Auto-Discovery

Instead of manually listing every model, its file path, and command parameters, you tell `llama-swap-config-autogen` where your models are stored.

### Expected Directory Layout

The tool scans the target directories looking for `.gguf` files. It expects the following depth configurations:

```text
<models_dir>/<model_name>/*.gguf
# Or with variants
<models_dir>/<model_name>/<variant_name>/*.gguf
```

> [!NOTE]
> If there is an extra leading category directory (e.g. `<models_dir>/Qwen/Qwen3-30B/...`), the first segment (`Qwen`) is automatically ignored when generating model IDs.

---

## 2. Parameterized Macros: Eliminating Numeric Duplications

When writing LLM command arguments, you often repeat flags with slight numeric differences (e.g., context lengths `--ctx-size 4096`, `--ctx-size 32768` or GPU layers `-ngl 25`, `-ngl 999`).

By using **positional arguments** `${1}`, `${2}`, etc. in your macros, you can define a single parameterized macro helper and pass parameters using a colon `:` and comma-separated values.

### Defining Parameterized Helpers

In your `macros` section, define helpers:

```yaml
macros:
  # Base helpers accepting parameters
  ctx: "--ctx-size ${1}"
  ngl: "--n-gpu-layers ${1}"
  cpu-offload: "--n-cpu-moe ${1} --threads ${2}"
```

### Referencing in Other Macros

You can call these parameterized macros inside other macros:

```yaml
macros:
  # Using defaults
  default-params: "${common-base} ${ngl:999} ${ctx:32768}"

  # Custom numeric overrides
  gemma-large-params: "${default-params} ${ngl:25} ${ctx:32768}"
  qwen-cpu-params: "${default-params} ${cpu-offload:12,16} ${ctx:65536}"
```

This prevents the explosion of utility macros like `context-short`, `context-medium`, `context-large`, etc.

---

## 3. Variant Presets with Implicit Argument Binding

Often, you want to generate multiple variants of a single discovered GGUF model (e.g., a "CPU-offload" version, a "Short context" version, or a "Fast inference" version).

Instead of repeating these variant definitions under every model, you can define **Variant Presets** and link them to model patterns using implicit variable binding.

### Step A: Define Presets with Placeholders

```yaml
variant_presets:
  # CPU-offload variant templates (expects ${cpu-params} to be bound)
  cpu:
    - suffix: ' (with CPU)'
      macro: '${cpu-params}'

  # Shared short context variant
  short-ctx:
    - suffix: ' (short ctx)'
      macro: 'short-ctx-params'
```

### Step B: Apply Presets to Model Patterns

In `model_patterns`, bind the model-specific parameters (like `cpu-params`) and list the presets to apply:

```yaml
model_patterns:
  # For Qwen3-30B, bind Qwen-specific CPU parameters and apply 'cpu' and 'short-ctx'
  qwen3-30b/instruct-2507:
    macro: qwen-context-large-params
    cpu-params: qwen-cpu-params      # Binds ${cpu-params} inside the 'cpu' preset
    variants: [cpu, short-ctx]       # Generates both variants automatically!

  # For Qwen3-Next-80B, bind BIG Qwen CPU parameters and apply 'cpu'
  qwen3-next-80b:
    macro: qwen-context-large-params
    cpu-params: qwen-cpu-params-big  # Binds BIG cpu params instead
    variants: [cpu]
```

### Benefits

- **No Duplicate Definitions**: The `cpu` template is defined only once globally.
- **Clean Syntax**: You just list the tags `[cpu, short-ctx]` under the model patterns.
- **Centralized Naming**: Suffix naming conventions are kept in `variant_presets`, ensuring consistency.

---

## 4. Model Labels with Multi-Pattern Lists

To make your llama-swap web UI look premium, you can decorate model names with visual tags (emojis). For example, `👁️` for vision models, `🎧` for audio, or `🔊` for TTS.

If you have multiple models requiring the same label (like many vision-capable models), you can group them into a single rule using a list pattern.

```yaml
model_labels:
  mmproj_default: ' 🌐'
  rules:
    # Match any of these substrings to attach ' 👁️'
    - pattern:
        - qwen-vl
        - qwen3-vl
        - glm-4.6v
        - gemma-3
        - gemma-4
        - small-3.2
      label: ' 👁️'
      requires_mmproj: true # Only label if mmproj is successfully attached

    - pattern: whisper
      label: ' 🎧'
```

This saves you from copying and pasting the same rule block for every multimodal model family.

---

## 5. Explicit Multi-Modal Projection (mmproj) Binding

When configuring multimodal (vision) models, you might want to share a single `mmproj` file among different variants of the same model family, even if the variants are stored in different directories (e.g., when a fine-tuned model doesn't ship with its own `mmproj` file).

Instead of manually maintaining absolute paths in a global `mmproj.overrides` section, you can explicitly attach an `mmproj` file directly inside the `model_patterns` mapping. You can specify:

- An absolute path,
- A relative path from the config file directory, or
- Just the **filename** of any mmproj file discovered during directory scanning (case-insensitive).

```yaml
model_patterns:
  # The fine-tuned variant doesn't have an mmproj file in its own directory.
  # We bind the mmproj file from the 'it' variant by simply writing its filename.
  gemma-4-31b:
    macro: gemma-4-32k-q8-off-params
    emit_base: false
    variants: [gemma-4-variants]
    mmproj: gemma-4-31b-mmproj-BF16.gguf
```

Using this pattern matches the target model family prefix (`gemma-4-31b`) and resolves the `mmproj` filename automatically to the scanned absolute path of the file, cleanly applying it to all matching variants.

---

## 6. Put It All Together: A Complete base.yaml Example

Here is a complete, real-world inspired `base.yaml` that uses all the advanced features discussed above:

```yaml
# 1. Scanning directories
models:
  - /opt/data/llm/models

vram_estimation: true
default_ttl: 300
healthCheckTimeout: 240
logLevel: warn
startPort: 9091

# 2. Multimodal attachment settings
mmproj:
  enabled: true
  auto_attach: true
  arg: --mmproj
  generate_no_mmproj_variant: false
  no_mmproj_suffix: ' (no mmproj)'
  overrides: {}

# 3. Model visual labels
model_labels:
  mmproj_default: ' 🌐'
  rules:
    - pattern: [qwen-vl, qwen3-vl, glm-4.6v, gemma-3, gemma-4]
      label: ' 👁️'
      requires_mmproj: true
    - pattern: whisper
      label: ' 🎧'

# 4. DRY parameterized macros
macros:
  binary: /app/llama-server
  common-base: --jinja --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
  default-params: ${common-base} ${ngl:999} ${ctx:32768}
  reasoning-base: ${default-params} --reasoning-format none

  # Parameterized macro helpers
  ctx: --ctx-size ${1}
  ngl: --n-gpu-layers ${1}
  cpu-offload: --n-cpu-moe ${1} --threads ${2}

  # Presets using helpers
  deepseek-r1-params: ${reasoning-base} --temp 0.7 --top-p 0.9
  qwen-cpu-params: ${default-params} ${cpu-offload:12,16} ${ctx:65536}
  qwen-cpu-params-big: ${default-params} ${ngl:15} ${cpu-offload:12,16} ${ctx:65536}
  short-ctx-params: ${default-params} ${ctx:4096}

# 5. Model configurations and variant presets mappings
model_patterns:
  r1-distill-qwen-32b:
    macro: deepseek-r1-params
    variants: [short-ctx]

  # Binds an mmproj file explicitly using its filename
  gemma-4-31b:
    macro: default-params
    variants: [short-ctx]
    mmproj: gemma-4-31b-mmproj-BF16.gguf

  qwen3-30b/instruct-2507:
    macro: default-params
    cpu-params: qwen-cpu-params
    variants: [cpu, short-ctx]

  qwen3-next-80b:
    macro: default-params
    cpu-params: qwen-cpu-params-big
    variants: [cpu]

variant_presets:
  cpu:
    - suffix: ' (with CPU)'
      macro: ${cpu-params}
  short-ctx:
    - suffix: ' (short ctx)'
      macro: short-ctx-params
```

Run the generator to produce your final `config.yaml`:

```bash
llama-swap-config-autogen generate --config base.yaml --output config.yaml
```

The resulting `config.yaml` will contain fully expanded commands, calculated VRAM labels (e.g., `[14.3 GB]`), unique ports starting from `9091`, and all variant models correctly registered for use!
