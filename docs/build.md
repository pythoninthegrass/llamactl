# Building llama-cpp-turboquant

## Prerequisites

```bash
brew install libomp
```

Apple Clang does not ship OpenMP. Without `libomp`, the GGML CPU backend
runs single-threaded for operations that fall back from Metal.

## Compile

```bash
cd ~/git
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cac

cmake -B build \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib

cmake --build build --config Release -j
```

Verify turbo types are available:

```bash
./build/bin/llama-server --help | grep turbo
```

## Symlink

```bash
mkdir -p ~/.local/bin
ln -s ~/git/llama-cpp-turboquant/build/bin/llama-server ~/.local/bin/llama-server
```

## Running manually

### With models.ini (router mode)

```bash
llama-server --host 0.0.0.0 --port 8080 --models-preset ~/models/models.ini
```

Router mode is experimental. The server registers models from the ini
file and from the HF cache, loading them on demand.

### Single model

```bash
llama-server --host 0.0.0.0 --port 8080 \
  -hf ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M \
  -ngl 99 -c 32768 --jinja
```

### TurboQuant KV cache compression

For models that tolerate symmetric quantization:

```bash
llama-server -m model.gguf -ctk turbo3 -ctv turbo3 -fa on
```
