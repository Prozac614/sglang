# AGENTS.md

## Cursor Cloud specific instructions

### Repository overview

SGLang is a high-performance LLM/multimodal serving framework. The repo is a monorepo with three main components:

| Component | Path | Language | Build |
|---|---|---|---|
| SGLang Runtime (SRT) | `python/` | Python | `pip install -e "python[dev]"` |
| SGL Model Gateway | `sgl-model-gateway/` | Rust | `cargo build` / `cargo test` |
| SGL Kernel | `sgl-kernel/` | C++/CUDA | Requires NVIDIA GPU |

### Environment caveats (no-GPU Cloud Agent)

- **No NVIDIA GPU available** in the Cloud Agent VM. The SGLang runtime (`python -m sglang.launch_server`) and sgl-kernel cannot run without a GPU. Many unit tests also fail to import due to CUDA initialization at module load time.
- The gateway (Rust binary and Python bindings) runs fully without GPU.
- **Clang is the default `cc`/`c++`** but cannot find `libstdc++` headers. Set `CC=gcc CXX=g++` when building Rust crates or C++ code.
- **`libstdc++.so` symlink** may be missing; ensure `/usr/lib/x86_64-linux-gnu/libstdc++.so` exists (symlink to `libstdc++.so.6`).
- **protoc** is needed for gRPC proto compilation in the gateway build. Install via `apt-get install protobuf-compiler libprotobuf-dev`.
- **uv** is installed at `/home/ubuntu/.local/bin/uv`. Add to `PATH` if needed: `export PATH="/home/ubuntu/.local/bin:$PATH"`.

### Lint

Lint tools configured via `.pre-commit-config.yaml`: `ruff` (F401/F821), `black`, `isort`, `codespell`, `clang-format`. Run individually or via `pre-commit run --all-files`.

### Tests

- **Python CPU-only unit tests**: `pytest test/registered/unit/server_args/ test/registered/unit/parser/ test/registered/unit/observability/ test/registered/unit/layers/ -v` (180+ pass on CPU).
- **Rust gateway tests**: `cd sgl-model-gateway && CC=gcc CXX=g++ PROTOC=/usr/bin/protoc cargo test` (91 pass; 1 Redis test fails without `redis-server` installed).
- **GPU-dependent tests** (most of `test/srt/`, `test/registered/unit/managers/`, `test/registered/unit/mem_cache/`): require NVIDIA GPU.

### Running services

- **Gateway**: `./sgl-model-gateway/target/debug/sgl-model-gateway launch --host 127.0.0.1 --port 30000 --worker-urls <backend_urls>`
- **SGLang Runtime** (requires GPU): `python -m sglang.launch_server --model-path <model> --port 8000`
- **Gateway Python bindings**: Built via `maturin build --manifest-path sgl-model-gateway/bindings/python/Cargo.toml --out dist` then `pip install dist/*.whl`.

### Key build env vars for the gateway

```bash
export CC=gcc
export CXX=g++
export PROTOC=/usr/bin/protoc
```
