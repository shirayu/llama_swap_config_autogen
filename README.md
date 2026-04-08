# Llama-Swap Config Autogen

![Header](./header.avif)

**Auto-discover GGUF models from directories for [llama-swap](https://github.com/mostlygeek/llama-swap) config**

Yes, llama-swap has macros. But you still need to manually list every model:

```yaml
# Even with macros, you manually write each model entry
macros:
  default-cmd: "${binary} -m ${model-path} --port ${PORT} ${params}"
models:
  "llama-2-7b": { cmd: "${default-cmd}", model-path: "/models/llama-2-7b.gguf" }
  "llama-2-13b": { cmd: "${default-cmd}", model-path: "/models/llama-2-13b.gguf" }
  "qwen-32b": { cmd: "${default-cmd}", model-path: "/models/qwen-32b.gguf" }
  # Still manually listing all models... 😵
```

This tool does the **model discovery** automatically:

```yaml
# Point to directories, get all models instantly
models: ["/opt/models", "/more/models"]
model_patterns: { "llama-2": "gpu-params", "qwen": "cpu-params" }
```

**The magic:** Scans directories → finds all `.gguf` files → applies pattern-based configs → generates complete llama-swap YAML.

## Get Started

### 1. Install

Use your favorite tool!

```bash
# With pip
pip install 'git+https://github.com/shirayu/llama_swap_config_autogen'

# With uv
uv tool install 'git+https://github.com/shirayu/llama_swap_config_autogen'
```

### 2. Generate a base file

```bash
llama-swap-config-autogen init --model /opt/llama.cpp/models --binary /opt/llama.cpp/bin/llama-server --output base.yaml
```

### 3. Generate the config file

Please edit `base.yaml` if you want.

```bash
llama-swap-config-autogen generate --config base.yaml --output config.yaml
```

The tool scans directories → finds all `.gguf` models → matches patterns → generates complete llama-swap config.

### Directory layout requirements

The generator now treats the directory structure under each entry in `models:` as the source of truth for model naming.

Supported layouts for each `models:` entry:

```text
<models_dir>/<model>/<file>.gguf
<models_dir>/<model>/<variant>/<file>.gguf
```

If a `models:` entry is instead organized with one extra leading category directory, that first segment is ignored automatically:

```text
<models_dir>/<family>/<model>/<file>.gguf
<models_dir>/<family>/<model>/<variant>/<file>.gguf
```

Examples:

```text
/opt/data/llm/models/Qwen/
  Qwen3-30B/
    Instruct-2507/
      unsloth_Qwen3-30B-A3B-Instruct-2507-GGUF_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf

/opt/data/llm/models/
  DeepSeek/
    DeepSeek-R1-Distill-Qwen-32B/
      Distill/
        unsloth_DeepSeek-R1-Distill-Qwen-32B-GGUF_DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf
  Qwen/
    Qwen3-30B/
      Instruct-2507/
        unsloth_Qwen3-30B-A3B-Instruct-2507-GGUF_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

Generated model IDs are built from the relative directory path plus the quantization suffix from the filename.
Generated `name` values also include the quantization suffix:

```text
model_id: qwen3-30b/instruct-2507:Q4_K_M
name: qwen3-30b/instruct-2507:Q4_K_M

model_id: deepseek-r1-distill-qwen-32b/distill:Q4_K_M
name: deepseek-r1-distill-qwen-32b/distill:Q4_K_M
```

Notes:

- `mmproj` auto-attach is resolved within the same directory as the model file.
- If the first directory segment is only a category such as `Qwen`, `Gemma`, or `DeepSeek`, it is ignored automatically.
- If a `.gguf` file is placed at an unexpected depth, generation fails with an error.
  Expected depth is `model` or `model/variant`, with an optional ignored leading family directory.
- A single `models:` entry must use one style consistently.
  Mixing `model/...` and `family/model/...` under the same root is treated as an error.
- Generated `name` values must be unique.
  Since quantization is included in `name`, different quantizations in the same directory are distinguished automatically.

### `base.yaml` spec

`base.yaml` defines model directories, macro templates, pattern-based macro selection, and optional variants.
For the full input format and behavior details, see [`spec.md`](./spec.md).
`mmproj` behavior (auto attach, overrides, and optional "no mmproj" variants) is also controlled in `base.yaml`.

Minimal example:

```yaml
models:
  - /opt/data/llm/models

macros:
  binary: /app/llama-server
  default-params: --jinja --flash-attn on --n-gpu-layers 999 --ctx-size 32768

model_patterns:
  qwen3: default-params
```

#### 4. Use it!

```bash
llama-swap --config ./config.yaml --watch-config -listen 0.0.0.0:9090
```

## Bonus

```bash
# Validation
llama-swap-config-autogen validate config.yaml
```

## License

Apache 2.0
