# llama-cpp-turboquant V100/SM70 Docker

Custom SM70 build of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) for Tesla V100.

## Image

- `ghcr.io/renne/llama-cpp-turboquant-v100-docker:tqp-v0.3.0-sm70`
- `ghcr.io/renne/llama-cpp-turboquant-v100-docker:latest`

## Build Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `CMAKE_CUDA_ARCHITECTURES` | `70` | V100/Volta only, no multi-arch bloat |
| `GGML_CUDA_FA_ALL_QUANTS` | `ON` | Enables turbo2/3/4 KV in flash attention VEC kernels |
| `GGML_CUDA` | `ON` | CUDA backend |
| `GGML_BACKEND_DL` | `ON` | Dynamic backend loading |
| `GGML_NATIVE` | `OFF` | Non-native build (explicit arch) |
| Base (build) | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | Full toolkit for compilation |
| Base (runtime) | `nvidia/cuda:12.8.1-runtime-ubuntu24.04` | Minimal runtime |

## Build Triggers

- Daily: 03:00 UTC
- Manual: `gh workflow run build.yml --repo renne/llama-cpp-turboquant-v100-docker -f turboquant_ref=tqp-v0.3.0`

## Usage

```bash
docker run --gpus all -p 8080:8080 \
  -v /mnt/models:/models:ro \
  ghcr.io/renne/llama-cpp-turboquant-v100-docker:latest \
  --model /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v turbo3 \
  --ctx-size 1310720 \
  --parallel 10 \
  --gpu-layers 999 \
  --cont-batching \
  --jinja
```

## Binaries

- `llama-server` — OpenAI-compatible HTTP server (ENTRYPOINT)
- `llama-bench` — Benchmark tool
- `llama-quantize` — Weight quantization (TQ4_1S, TQ3_1S conversion)

## TurboQuant KV Cache Types

| Type | Bits | Compression | Notes |
|------|------|-------------|-------|
| `turbo4` | ~4.5 | Lightest | Start here for new models |
| `turbo3` | ~3.5 | 4.6x at <1.5% PPL loss | Recommended default |
| `turbo2` | ~2.0 | Aggressive | Boundary V auto-engages |

Recommended asymmetric config: `--cache-type-k q8_0 --cache-type-v turbo3`
