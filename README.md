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

`llama-swap-config-autogen` automates model discovery, wraps models in visual labels, resolves VRAM estimation dynamically, and collapses variations using **parameterized macros** and **variant presets**:

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

**The magic:** Scans directories → finds all GGUF models → applies template-based rules → outputs complete, ready-to-run llama-swap configs.

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

When `vram_estimation: true` is set, the generator reads metadata headers directly from discovered GGUFs to calculate required VRAM (based on active GPU offload layers `-ngl` and context length `-c` resolved from your macros) and appends it to model display names:

```text
name: qwen3-30b/instruct-2507:Q4_K_M [18.8 GB]
```

Metadata is locally cached under `~/.cache/llama_swap_config_autogen/gguf_metadata.json` and automatically invalidated when GGUF files change.

---

## Advanced Configurations (DRY Concept)

To keep your `base.yaml` short and free from copy-paste duplications, the tool supports three advanced features:

1. **Parameterized Macros**: Declare positional templates like `${ngl:999}` or `${ctx:32768}` to prevent creating distinct macros for every context size.
2. **Variant Presets**: Define variant templates (e.g. CPU offload) once, and bind arguments at the model pattern level using tags (`variants: [cpu, short-ctx]`).
3. **Multi-Pattern Labels**: Attach visual tags (like `👁️`, `🎧`) to multiple model matches using lists of substrings in rules.

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
