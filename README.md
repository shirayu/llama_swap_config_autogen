# Llama-Swap Config Autogen

![Header](./header.avif)

**Auto-discover GGUF models from directories and generate [llama-swap](https://github.com/mostlygeek/llama-swap) configs with zero redundancy**

Yes, llama-swap supports macros, but writing and maintaining flat yaml entries for dozens of model variations and numeric flag differences is still painful:

```yaml
# ❌ The manual/repetitive way:
macros:
  default-cmd: "${binary} -m ${model-path} --port ${PORT} ${params}"
models:
  "llama-2-7b": { cmd: "${default-cmd}", model-path: "/models/llama-2-7b.gguf" }
  "llama-2-13b": { cmd: "${default-cmd}", model-path: "/models/llama-2-13b.gguf" }
  "qwen-32b": { cmd: "${default-cmd}", model-path: "/models/qwen-32b.gguf" }
  # Still manually listing all files and quantizations... 😵
```

`llama-swap-config-autogen` automates model discovery, exposes model metadata structurally, resolves VRAM estimation dynamically, and collapses variations using **parameterized macros** and **variant presets**:

```yaml
#  The automated, DRY way in base.yaml:
models: ["/opt/models"]

variant_presets:
  cpu: [{ suffix: ' (with CPU)', macro: '${cpu-params}' }]

model_patterns:
  qwen3:
    macro: default-params
    cpu-params: qwen-cpu-params
    variants: [cpu]
```

**The magic:** Scans directories → finds all GGUF models → applies template-based rules → outputs complete, ready-to-run llama-swap configs, each with an accurate `capabilities` block (context length, vision input, tool calling) auto-derived from your commands and the GGUF files themselves — no manual bookkeeping.

---

## Get Started

### 1. Install

Install via your preferred Python installer:

```bash
# With pip
pip install 'git+https://github.com/shirayu/llama_swap_config_autogen'

# With uv
uv tool install 'git+https://github.com/shirayu/llama_swap_config_autogen'
```

### 2. Generate a base configuration

Create a baseline `base.yaml` that template-scans your model directory:

```bash
llama-swap-config-autogen init --model /opt/llama.cpp/models --binary /opt/llama.cpp/bin/llama-server --output base.yaml
```

### 3. Generate the llama-swap config

Compile the human-friendly `base.yaml` rules into the machine-ready llama-swap `config.yaml`:

```bash
llama-swap-config-autogen generate --config base.yaml --output config.yaml
```

Then run llama-swap:

```bash
llama-swap --config ./config.yaml --watch-config -listen 0.0.0.0:9090
```

---

## Key Features

### 📂 Directory Layout & Auto-Discovery

The generator automatically discovers `.gguf` files under each entry in `models:`. It supports standard layouts:

```text
<models_dir>/<model_name>/*.gguf
<models_dir>/<model_name>/<variant_name>/*.gguf
```

*(Optional parent directories like `<models_dir>/Family/model/*.gguf` are supported and the category prefix is automatically ignored during model ID assignment).*

### 💾 Dynamic VRAM Estimation

When `vram_estimation: true` is set, the generator reads metadata headers directly from discovered GGUFs to calculate required VRAM (based on active GPU offload layers `-ngl` and context length `-c` resolved from your macros) and emits it as structured model metadata:

```text
name: qwen3-30b/instruct-2507:Q4_K_M
metadata:
  model_family: qwen3-30b
  model_size_gib: 18.8
```

Metadata is locally cached under `~/.cache/llama_swap_config_autogen/gguf_metadata.json` and automatically invalidated when GGUF files change.

Two more fields are auto-derived into `metadata`, so "thinking off" no longer has to be encoded only in a display-name suffix:

- `reasoning: "off"`: added whenever the expanded launch command contains `--reasoning off` (e.g. a `thinking-off` variant). `reasoning: "on"` is added when GGUF metadata reports reasoning support and the command does not disable it. The `off` value takes precedence. The command-only `off` detection works even with `vram_estimation: false`.
- `reasoning_supported`: `true`/`false` reflecting whether the GGUF's chat template indicates the model architecture supports reasoning/thinking mode at all (independent of whether a given variant launches with it off). Unlike `tools` (below), this is always emitted (not omitted on `false`) whenever GGUF headers are read — set `vram_estimation: true`, or set `read_gguf_metadata: true` if you want this detection without paying for VRAM estimation.

```text
name: qwen3-30b/instruct-2507:Q4_K_M
metadata:
  model_family: qwen3-30b
  model_size_gib: 18.8
  reasoning_supported: true
  reasoning: "on"
```

### 📏 Model Capabilities

llama-swap supports a [`capabilities`](https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml) block per model (`context`, `in`, `out`, `tools`, `reranker`) that it exposes via `/v1/models`, but this generator used to leave it out entirely — clients had no reliable way to know a model's real context length, modality, or tool-calling support (see [mostlygeek/llama-swap#999](https://github.com/mostlygeek/llama-swap/issues/999) for a client-side symptom of this same gap). Every generated model entry now gets an auto-derived `capabilities` block, so llama-swap's `/v1/models` reflects each model's real capabilities without manual configuration:

- `context`: resolved from `-c`/`--ctx-size` in the expanded command (falling back to GGUF metadata when `vram_estimation: true` or `read_gguf_metadata: true`).
- `in`: `[text, image]` when an `mmproj` is attached to the entry, otherwise `[text]`.
- `tools`: `true` when `vram_estimation: true` or `read_gguf_metadata: true` and the GGUF's chat template references tool calling; omitted otherwise.

Set `read_gguf_metadata: true` if you want GGUF-header-derived fields (`tools`, `metadata.reasoning_supported`, context fallback) without also paying for VRAM estimation — `vram_estimation: true` implies it.

This lands directly in `config.yaml`, so clients like Open WebUI show accurate context limits and correctly gate vision/tool-calling UI per model — no hand-maintained metadata to keep in sync:

```yaml
# Generated config.yaml
models:
  "qwen3-30b/instruct-2507:Q4_K_M":
    cmd: ...
    name: "qwen3-30b/instruct-2507:Q4_K_M"
    metadata:
      model_family: qwen3-30b
      model_size_gib: 18.8
    capabilities:
      context: 32768
      in: [text]
      tools: true
```

You can override any of these, and declare `out`/`reranker` (which aren't auto-derived), per `model_patterns` entry:

```yaml
model_patterns:
  qwen3-30b:
    macro: default-params
    capabilities:
      out: [text]
      reranker: false
```

---

## Advanced Configurations (DRY Concept)

To keep your `base.yaml` short and free from copy-paste duplications, the tool supports three advanced features:

1. **Parameterized Macros**: Declare positional templates like `${ngl:999}` or `${ctx:32768}` to prevent creating distinct macros for every context size.
2. **Variant Presets**: Define variant templates (e.g. CPU offload) once, and bind arguments at the model pattern level using tags (`variants: [cpu, short-ctx]`).
3. **Structured Model Metadata**: Expose model family, VRAM estimates, and modality capabilities through llama-swap metadata instead of display-name decorations.

For a detailed step-by-step guide with examples, see the [**`tutorial.md`**](./tutorial.md).
For the complete technical file format definition, see the [**`spec.md`**](./spec.md).

### Minimal Example with Parameterized Macros

```yaml
models:
  - /opt/data/llm/models

vram_estimation: true

macros:
  binary: /app/llama-server
  common-base: --jinja --flash-attn on
  # Helper macros accepting positional parameters
  ngl: --n-gpu-layers ${1}
  ctx: --ctx-size ${1}
  # Call helpers with parameters
  default-params: ${common-base} ${ngl:999} ${ctx:32768}

model_patterns:
  qwen3: default-params
```

---

## Utility Commands

```bash
# Verify the generated llama-swap configuration for errors
llama-swap-config-autogen validate config.yaml

# Run generation with detailed logs and VRAM calculation traces
llama-swap-config-autogen generate --config base.yaml --verbose
```

## License

Apache 2.0
