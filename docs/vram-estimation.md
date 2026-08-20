# VRAM Estimation

`llama-swap-config-autogen` can annotate every generated model entry with an estimated VRAM footprint, computed by
llama.cpp's own [`fit-params --fit-print on`](https://github.com/ggml-org/llama.cpp) tool rather than a hand-rolled
formula. Since it asks llama.cpp itself, it accounts for compute buffers, cache quantization, and CPU-offload flags
the way the real `llama-server` process would.

## Enabling it

Two things are required:

1. `vram_estimation: true` in `base.yaml`.
2. `--llama-bin <command>` passed to `generate`.

```bash
llama-swap-config-autogen generate --config base.yaml --llama-bin /opt/llama.cpp/bin/llama
```

`--llama-bin` accepts any command that ends up running a `fit-params`-capable llama.cpp binary, split with `shlex`.
This lets you point at a binary running inside a container:

```bash
llama-swap-config-autogen generate --config base.yaml \
  --llama-bin "podman container exec llama-swap /app/llama"
```

Because `fit-params` loads (without allocating) the actual model file, estimation only works on a machine that can
reach both the model files and the `fit-params` binary. If `vram_estimation: true` is set but `--llama-bin` is
omitted, generation still succeeds — VRAM estimation is just skipped, with a warning on stderr.

## What gets emitted

```yaml
name: qwen3-30b/instruct-2507:Q4_K_M
metadata:
  model_family: qwen3-30b
  estimated_vram_bytes: 20182171238
  file_size_bytes: 18933312716
```

`estimated_vram_bytes` sums the model, context, and compute buffer sizes `fit-params` reports for each GPU device,
using each model's resolved `-ngl`, `-c`, `--cache-type-k`/`-v`, and CPU-offload flags (`-ot`, `--cpu-moe`,
`--n-cpu-moe`) — all forwarded to `fit-params` as-is so it simulates the real tensor placement.

## Path mapping for containerized runtimes

If the runtime sees model files under a different path than the one scanned under `models:` (e.g. a container
bind-mount), set `path_prefix_map` in `base.yaml`. It rewrites host paths to runtime paths using the longest
matching prefix, and applies both to the `-m`/`--mmproj` arguments in generated commands and to the path passed to
`fit-params`:

```yaml
path_prefix_map:
  /opt/data/llm/models/: /models/
```

See [`containerized-setup.md`](./containerized-setup.md) for the full picture of running llama.cpp/llama-swap in a
container, including `init --binary` and how to launch llama-swap itself.

## Caching

- VRAM results are cached under `~/.cache/llama_swap_config_autogen/fit_params_vram.json`, keyed by each model
  file's mtime/size plus the resolved `-ngl`/`-c`/cache-type/CPU-offload arguments. Re-running `generate` only
  invokes `fit-params` again for models whose file or relevant flags changed.
- GGUF header metadata used for capability detection (`tools`, `reasoning_supported`, expert counts, etc.) is
  cached separately under `~/.cache/llama_swap_config_autogen/gguf_metadata.json`.

## Related auto-derived metadata

Enabling GGUF header reads (`vram_estimation: true`, or `read_gguf_metadata: true` if you only want detection
without paying for VRAM estimation) also derives these `metadata` fields:

- `reasoning_supported`: `true`/`false` reflecting whether the GGUF's chat template indicates the model
  architecture supports reasoning/thinking mode. Describes model capability, not a per-request setting. Always
  emitted (not omitted on `false`).
- `file_size_bytes`: the GGUF file's actual size on disk, in bytes.
- `expert_count` / `expert_used_count`: total and active experts, emitted only for mixture-of-experts models.
- `repo_url`: source repository URL embedded in the GGUF (`general.repo_url` / `general.source.repo_url`), when set.
- `license`: license name embedded in the GGUF (`general.license`), when set.

```yaml
name: qwen3-30b/instruct-2507:Q4_K_M
metadata:
  model_family: qwen3-30b
  estimated_vram_bytes: 20182171238
  file_size_bytes: 18933312716
  reasoning_supported: true
  expert_count: 128
  expert_used_count: 8
  repo_url: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
  license: apache-2.0
```

### Sidecar metadata (`<model>.json`)

Drop a `<model>.json` next to a GGUF (same basename, e.g. `llama3-Q4_K_M.gguf` + `llama3-Q4_K_M.json`) to
hand-author or override any metadata field. Its contents are merged into `metadata` last, so they take precedence
over anything auto-derived from the GGUF headers:

```json
{
  "notes": "fine-tuned in-house, do not redistribute",
  "license": "custom"
}
```
