# Running with a Containerized llama.cpp

The [Get Started](../README.md#get-started) steps assume `llama-server` and `llama` run natively on the same
machine you run `llama-swap-config-autogen` on. If llama.cpp (and/or llama-swap) instead runs inside a container
(Podman, Docker, ...), a few things differ. This page collects them in one place.

## 1. `init --binary`

Use the path `llama-server` has *inside* the container, not the host path:

```bash
llama-swap-config-autogen init --model /opt/llama.cpp/models --binary /app/llama-server --output base.yaml
```

This becomes the `${binary}` macro, used verbatim in every generated `cmd:`, which llama-swap will execute inside
that same container.

## 2. `models:` path vs. what the container sees

`llama-swap-config-autogen generate` scans `models:` directories on the machine it runs on. If that's the host, but
the container mounts those files at a different path, set `path_prefix_map` in `base.yaml` so the generated `-m`
argument (and any `fit-params` VRAM lookup) uses the container-visible path instead of the host path:

```yaml
models:
  - /opt/data/llm/models   # host path, scanned by the generator

path_prefix_map:
  /opt/data/llm/models/: /models/   # path the container actually sees
```

See [`spec.md` §3.9](./spec.md#39-path_prefix_map-optional) for the exact matching rules.

## 3. `generate --llama-bin` for VRAM estimation

If you want VRAM estimates (`vram_estimation: true`), `--llama-bin` needs to reach a `fit-params`-capable llama.cpp
binary. When that binary only exists inside a running container, point at it through `exec`:

```bash
llama-swap-config-autogen generate --config base.yaml --output config.yaml \
  --llama-bin "podman container exec llama-swap /app/llama"
# or with Docker:
  --llama-bin "docker exec llama-swap /app/llama"
```

`--llama-bin` is split with `shlex`, so any command that ultimately runs the binary works — it doesn't have to be a
bare path. See [`vram-estimation.md`](./vram-estimation.md) for details.

## 4. Running llama-swap itself

Run llama-swap the way you already run the rest of your stack — as a container with the generated `config.yaml`
and your models directory mounted in, e.g.:

```bash
podman run -d --name llama-swap \
  -v ./config.yaml:/app/config.yaml:ro \
  -v /opt/data/llm/models:/models:ro \
  -p 9090:9090 \
  ghcr.io/mostlygeek/llama-swap:cuda \
  --config /app/config.yaml --watch-config --listen 0.0.0.0:9090
```

Whatever image/tag and GPU flags you need are specific to your setup — see llama-swap's own docs for those.
