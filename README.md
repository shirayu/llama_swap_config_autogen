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
