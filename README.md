# Llama-Swap Config Autogen

![Header](./header.avif)

**Point it at your GGUF folder, get a complete [llama-swap](https://github.com/mostlygeek/llama-swap) config back — no per-model YAML entries to write or maintain.**

- 🔍 **Auto-discovers models.** Drop files into `<models_dir>/<model>/*.gguf` and they show up in the config on the next run — no manual model list to edit.
- 🧬 **DRY by design.** One macro + one pattern rule covers every quantization and variant of a model family, instead of a copy-pasted YAML block per file.
- 💾 **Real VRAM numbers, not guesses.** Optionally asks llama.cpp's own `fit-params` tool how much VRAM each model actually needs, so you can plan what fits together.
- 📏 **Accurate `capabilities` for free.** Context length, vision input, tool-calling support — derived from your commands and the GGUF headers themselves, so clients like Open WebUI show the truth without you maintaining it by hand.

```yaml
# ❌ The manual way: one block per file, repeated for every quantization
models:
  "llama-2-7b": { cmd: "${default-cmd}", model-path: "/models/llama-2-7b.gguf" }
  "llama-2-13b": { cmd: "${default-cmd}", model-path: "/models/llama-2-13b.gguf" }
  "qwen-32b": { cmd: "${default-cmd}", model-path: "/models/qwen-32b.gguf" }
  # ...and every other file you own, forever kept in sync by hand 😵
```

```yaml
# ✅ The generated way: describe the rule once in base.yaml, in full
models: ["/opt/models"]

variant_presets:
  cpu: [{ suffix: ' (with CPU)', macro: '${cpu-params}' }]

model_patterns:
  qwen3:
    macro: default-params
    cpu-params: qwen-cpu-params
    variants: [cpu]
```

Every `.gguf` under `/opt/models` that matches a pattern gets a full config entry — quantization, variants, `capabilities`, VRAM metadata — generated for it automatically.

---

## Get Started

### 1. Install

```bash
# With pip
pip install 'git+https://github.com/shirayu/llama_swap_config_autogen'

# With uv
uv tool install 'git+https://github.com/shirayu/llama_swap_config_autogen'
```

### 2. Generate a base configuration

Scans your model directory and writes a starter `base.yaml`. `--binary` is the `llama-server` path that ends up in every generated command — use whatever path *runs* `llama-server` for you (a local binary, or a path inside a container image):

```bash
llama-swap-config-autogen init --model /opt/llama.cpp/models --binary /opt/llama.cpp/bin/llama-server --output base.yaml
```

### 3. Generate the llama-swap config

Compiles `base.yaml` into the machine-ready `config.yaml`, with `capabilities` auto-derived per model:

```bash
llama-swap-config-autogen generate --config base.yaml --output config.yaml
```

### 4. Run llama-swap

```bash
llama-swap --config ./config.yaml --watch-config -listen 0.0.0.0:9090
```

That's it — every model under your directory is now served. Re-run step 3 whenever you add or remove `.gguf` files.

**Want VRAM estimates too?** Add `vram_estimation: true` to `base.yaml` and pass `--llama-bin <path-to-llama.cpp-binary>` to `generate`. Details: [`docs/vram-estimation.md`](./docs/vram-estimation.md).

**Running llama.cpp/llama-swap in a container (Podman, Docker, ...) instead of natively?** Steps 2–4 above need small adjustments (binary path, model path mapping, `--llama-bin`, how you launch llama-swap) — see [`docs/containerized-setup.md`](./docs/containerized-setup.md).

---

## Learn More

| Doc | What it covers |
| --- | --- |
| [`docs/tutorial.md`](./docs/tutorial.md) | Step-by-step guide: parameterized macros, variant presets, mmproj binding, a full worked example. |
| [`docs/spec.md`](./docs/spec.md) | Complete `base.yaml` field reference and generation rules. |
| [`docs/vram-estimation.md`](./docs/vram-estimation.md) | Enabling VRAM estimation, `path_prefix_map`, caching, auto-derived metadata fields. |
| [`docs/capabilities.md`](./docs/capabilities.md) | How the `capabilities` block (`context`, `in`, `tools`, ...) is derived and how to override it. |
| [`docs/containerized-setup.md`](./docs/containerized-setup.md) | Running llama.cpp/llama-swap in a container instead of natively: binary paths, `path_prefix_map`, `--llama-bin`. |

### Directory layout

Models must sit under `models:` directories in one of these layouts:

```text
<models_dir>/<model_name>/*.gguf
<models_dir>/<model_name>/<variant_name>/*.gguf
```

An optional leading family directory (`<models_dir>/Family/model/*.gguf`) is also supported and ignored for naming.

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
