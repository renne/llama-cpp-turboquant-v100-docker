ARG CUDA_VERSION=12.8.1
ARG GCC_VERSION=14
ARG TURBOQUANT_REF=tqp-v0.3.0

# ─── Build stage ───────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ARG GCC_VERSION
ARG TURBOQUANT_REF

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} \
        g++-${GCC_VERSION} \
        build-essential \
        cmake \
        python3 \
        git \
        libssl-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION}
ENV CXX=g++-${GCC_VERSION}
ENV CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /app

# Clone TurboQuant+ fork (pinned to release tag)
RUN git clone --depth 1 --branch ${TURBOQUANT_REF} \
        https://github.com/TheTom/llama-cpp-turboquant.git /app

# Build: SM70 (V100) only + all FA quant instances for turbo KV
# GGML_CUDA_FA_ALL_QUANTS=ON compiles all fattn-vec template instances,
# enabling turbo2/turbo3/turbo4 KV types in flash attention VEC path.
# Without it, turbo KV works but NOT with FA VEC kernels.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        . && \
    cmake --build build --config Release -j$(nproc)

# Collect shared libraries (ggml backends) — matches upstream cuda.Dockerfile pattern
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

# Collect binaries
RUN mkdir -p /app/full && \
    cp build/bin/llama-server /app/full/ && \
    cp build/bin/llama-bench /app/full/ && \
    (cp build/bin/llama-quantize /app/full/ 2>/dev/null || true)

# ─── Runtime stage (server only) ──────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS server

ARG TURBOQUANT_REF

LABEL maintainer="renne"
LABEL description="TheTom/llama-cpp-turboquant V100/SM70 with all FA quant instances"
LABEL turboquant.ref="${TURBOQUANT_REF}"
LABEL target.arch="sm_70"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/ /app

ENV LD_LIBRARY_PATH=/app
ENV LLAMA_ARG_HOST=0.0.0.0

WORKDIR /app

HEALTHCHECK --interval=15s --timeout=10s --retries=20 --start-period=120s \
    CMD ["curl", "-sf", "http://localhost:8080/health"]

ENTRYPOINT ["/app/llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
