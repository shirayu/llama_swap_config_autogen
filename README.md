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

Want VRAM estimates in the output? Add `--llama-bin <path-to-llama.cpp-binary>` and set `vram_estimation: true` in `base.yaml` — see [`docs/vram-estimation.md`](./docs/vram-estimation.md).

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

With `vram_estimation: true` and `generate --llama-bin <command>`, each model entry gets an `estimated_vram_bytes` figure computed by llama.cpp's own `fit-params` tool — not a hand-rolled formula — so it reflects real compute buffers, cache quantization, and CPU-offload flags:

```text
name: qwen3-30b/instruct-2507:Q4_K_M
metadata:
  model_family: qwen3-30b
  estimated_vram_bytes: 20182171238
  file_size_bytes: 18933312716
  reasoning_supported: true
```

See [**`docs/vram-estimation.md`**](./docs/vram-estimation.md) for enabling it (including containerized `--llama-bin` setups), the `path_prefix_map` option, result caching, and the full list of auto-derived metadata fields (`reasoning_supported`, `expert_count`, `repo_url`, `license`, sidecar `<model>.json` overrides).

### 📏 Model Capabilities

llama-swap supports a [`capabilities`](https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml) block per model (`context`, `in`, `out`, `tools`, `reranker`) that it exposes via `/v1/models`, but this generator used to leave it out entirely — clients had no reliable way to know a model's real context length, modality, or tool-calling support (see [mostlygeek/llama-swap#999](https://github.com/mostlygeek/llama-swap/issues/999) for a client-side symptom of this same gap). Every generated model entry now gets an auto-derived `capabilities` block, so llama-swap's `/v1/models` reflects each model's real capabilities without manual configuration:

- `context`: resolved from `-c`/`--ctx-size` in the expanded command, falling back to GGUF metadata.
- `in`: `[text, image]` when an `mmproj` is attached to the entry, otherwise `[text]`.
- `tools`: `true` when the GGUF's chat template references tool calling; omitted otherwise.

The GGUF-derived parts of this (context fallback, `tools`, plus `metadata.reasoning_supported`) require GGUF headers to be read, which happens automatically when `vram_estimation: true`. Set `read_gguf_metadata: true` instead if you want them without also paying for VRAM estimation.

This lands directly in `config.yaml`, so clients like Open WebUI show accurate context limits and correctly gate vision/tool-calling UI per model — no hand-maintained metadata to keep in sync:

```yaml
# Generated config.yaml
models:
  "qwen3-30b/instruct-2507:Q4_K_M":
    cmd: ...
    name: "qwen3-30b/instruct-2507:Q4_K_M"
    metadata:
      model_family: qwen3-30b
      estimated_vram_bytes: 20182171238
      file_size_bytes: 18933312716
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

For a detailed step-by-step guide with examples, see [**`docs/tutorial.md`**](./docs/tutorial.md).
For the complete technical file format definition, see [**`docs/spec.md`**](./docs/spec.md).

### Minimal Example with Parameterized Macros

```yaml
models:
  - /opt/data/llm/models

vram_estimation: true # requires `generate --llama-bin <command>`, see docs/vram-estimation.md

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
