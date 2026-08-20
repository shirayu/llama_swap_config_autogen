# Model Capabilities

llama-swap supports a [`capabilities`](https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml) block
per model (`context`, `in`, `out`, `tools`, `reranker`) that it exposes via `/v1/models`, but hand-authoring one for
every generated entry defeats the point of auto-discovery — clients would have no reliable way to know a model's
real context length, modality, or tool-calling support otherwise (see
[mostlygeek/llama-swap#999](https://github.com/mostlygeek/llama-swap/issues/999) for a client-side symptom of this
gap). Every generated model entry gets an auto-derived `capabilities` block instead:

- `context`: resolved from `-c`/`--ctx-size` in the expanded command, falling back to GGUF metadata.
- `in`: `[text, image]` when an `mmproj` is attached to the entry, otherwise `[text]`.
- `tools`: `true` when the GGUF's chat template references tool calling; omitted otherwise.

The GGUF-derived parts of this (context fallback, `tools`, plus `metadata.reasoning_supported`) require GGUF headers
to be read, which happens automatically when `vram_estimation: true`. Set `read_gguf_metadata: true` instead if you
want them without also paying for VRAM estimation (see [`vram-estimation.md`](./vram-estimation.md)).

This lands directly in `config.yaml`, so clients like Open WebUI show accurate context limits and correctly gate
vision/tool-calling UI per model — no hand-maintained metadata to keep in sync:

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

## Overriding or extending

Override any auto-derived field, and declare `out`/`reranker` (which aren't auto-derived), per `model_patterns`
entry:

```yaml
model_patterns:
  qwen3-30b:
    macro: default-params
    capabilities:
      out: [text]
      reranker: false
```

Audio models should declare their modalities explicitly, since they can't be inferred from the GGUF or command:

```yaml
model_patterns:
  whisper:
    macro: default-params
    capabilities:
      in: [audio]
      out: [text]
```
