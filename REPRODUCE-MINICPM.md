# Reproduce the custom managed run

## Pinned builds

Start in the root of this recipe repository. Prepare separate native and Python checkouts; these commands do not launch a model:

```sh
RECIPE_ROOT="$PWD"
git clone https://github.com/BobClawblaw/TensorFold.git tensorfold-native-src
git -C tensorfold-native-src checkout --detach bfcc84ab22669fd92586572d5e57ad10a5eea716
git -C tensorfold-native-src apply --check "$RECIPE_ROOT/source/minicpm-native.patch"
git -C tensorfold-native-src apply "$RECIPE_ROOT/source/minicpm-native.patch"
git clone https://github.com/jkuepker/TensorFold.git tensorfold-python-src
git -C tensorfold-python-src checkout --detach 93fcf31c5bd861e5edb98ebf89d40e74b60b2d22
git -C tensorfold-python-src apply --check "$RECIPE_ROOT/source/minicpm-python.patch"
git -C tensorfold-python-src apply "$RECIPE_ROOT/source/minicpm-python.patch"
```

Native benchmark controller: BobClawblaw/TensorFold `bfcc84ab22669fd92586572d5e57ad10a5eea716` plus source/minicpm-native.patch. Build with exact Zig 0.17.0, CUDA 13.0.88, SM120:

```sh
cd tensorfold-native-src
python3 -m ziglang build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -j2
```

The patch's build.zig.zon supplies `1.0.12+frontier-b4`; do not replace it with a release-version flag. Output: zig-out/native/bin/tensorfold-native. Measured SHA256: `50b7b74426ac40f732df6ceabd3345da72fda2d2f90d939ae4a644a706967600`. The native benchmark suite, SSE/timing code, hardware probe and checkpoint digest code are byte-identical to the pinned upstream files; hashes are in evidence/unchanged-protocol-files.json. The patch adds only explicit MiniCPM managed launch/runtime reporting and diagnostic retention.

Python engine: jkuepker/TensorFold `93fcf31c5bd861e5edb98ebf89d40e74b60b2d22` plus source/minicpm-python.patch. Use Python 3.12.3, torch 2.11.0+cu130 and Triton 3.6.0. The measured cached runtime image ID is in recipes/minicpm.json; compiled image/binary artifacts are not distributed. Install the patched engine in a separate virtual environment or put its src directory first on PYTHONPATH. Do not install it over the native controller executable. The Python patch gives it version `0.6.6+mini-managed2`, supports reporting an actual OS-selected port, and disables MiniCPM's private prompt-cache lookup/storage when the benchmark adapter requests it. The old prompt-cache CLI flag alone is insufficient on that backend. No CUDA arithmetic/kernel code changes.

For each checkout, detach at the exact public base commit and apply the corresponding patch using `git apply --check` then `git apply`. Both complete patches were checked on their pinned bases and the resulting files matched the measured worktrees. Use the source tree's unchanged CUDA extension build flags and set TORCH_CUDA_ARCH_LIST=12.0. The first start/request may compile extensions; keep a writable runtime cache. The original shared cache had root-owned subdirectories, so this campaign copied its ~107 MB small kernel caches into an isolated writable directory. The benchmark's excluded warm-ups handle cold capture; do not alter measured request times.

## Public offline checkpoints

Target: kuepker/MiniCPM-V-4.7-35B-A3B-MLX-4bit-TensorFold at `0aee82655c3bdb7135099cd368af7e92235d066f`, including config/tokenizer and shards. MTP sidecar: TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP at `f84b054c5677b6e59bdb969ae61915b0983d68bb`, including mtp-4bit.safetensors. Model file digests come from the actual native resolver, not user-entered metadata. Keep both snapshots pinned in the public Hugging Face cache with refs/main set to these exact revisions; the native CLI accepts public model IDs and resolves their cached snapshots, rather than a revision launch flag. Do not rename a local directory to a public ID or modify the original model configuration. Any needed downloads must share an aggregate cap of 450 Mbit/s; existing assets were reused for this campaign.

## Adapter and launch

Copy source/minicpm-adapter.py to an executable path. Its shebang is `/opt/venv/bin/python` in the measured container; for a portable installation set it to the actual patched Python environment's interpreter, and verify `adapter.py --facts` reports the real installed runtime. A different environment needs its own new benchmark receipt. Exact measured critical dependency versions are retained in source/minicpm-dependency-versions.json; install the pinned source's remaining declared dependencies as well. This records the tested environment, not a promise that a different rebuild produces an identical binary. The adapter imports the actual engine version and Python version; it does not read arbitrary version strings from environment variables. Its pinned snapshot checks are deliberate. It maps the explicitly resolved public MTP sidecar to the fork's TF_QWEN36_MTP loader while declaring that sidecar in the receipt. Serial mode passes no drafter. It binds the HTTP server itself at a genuine ephemeral loopback port; there is no pre-existing server or metadata-upgraded attached run.

Choose a free GPU slot and acquire the shared testing lock before executing. Set HF_HUB_CACHE and PYTHONPATH to the prepared cache/engine locations. The measured container used 8 CPUs, 24 GB host-memory limit, no network, UID 1000 and compute,utility driver capabilities. Additional environment is recorded in recipes/minicpm.json; use writable per-installation Torch/Triton caches.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
export TENSORFOLD_CUDA_MEMORY_LIMIT_GB=24 TENSORFOLD_MEMORY_RESERVE_GIB=2
export TF_MINICPM_MANAGED_ADAPTER=/path/to/executable/adapter.py
export TF_MINICPM_MTP_DRAFTS=6 TF_MINICPM_CONFIDENCE=0.7 TF_MINICPM_KEEP_LOGS=1
export TF_BENCHMARK_NO_PREFIX_REUSE=1
/path/to/custom/tensorfold-native benchmark \
  kuepker/MiniCPM-V-4.7-35B-A3B-MLX-4bit-TensorFold --backend cuda \
  --drafter TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP \
  --context 2048 --output new-benchmark.json
```

Replace example paths with real paths. The parent resolves the target/head, detects hardware in the same local container, obtains runtime facts from the adapter, starts its own Python serve child with parallel=1 and cache/snapshots disabled, runs the untouched standard suite and reaps that child. The Python engine performs CUDA inference; the native binary is the benchmark client/supervisor. Sidecar MTP is verified by the MiniCPM target; it is not an unrelated external drafter. Confidence is 0.7. The family wrapper now explicitly forwards this value to the MTP engine; the original default path used 0.3. All standard prompts, seeds, temperatures, top-k/top-p, thinking/EOS settings and delivery timing are unchanged. Preserve source patches and version labels with any public recipe.

Use a sampled whole-device growth stop at 24,000 MiB in addition to the 24 GiB admission setting, leaving margin below the owner's 28 decimal GB additional-memory limit. Polling is not a hard allocator quota. Keep desktop/display active and the stock 200 W cap for comparable runs. Stop only the server/container you started, and restore no model. Never run a full-model Nsight replay under this memory budget. Retained server logs can contain local paths; they are not part of this public package.

## Published result and variant

The primary depth6/confidence0.7 receipt is published under Maxrio9; see publication.json. The code-oriented alternative sets TF_MINICPM_MTP_DRAFTS=5, with the same source and confidence. It is measured and checked but not separately submitted. The native controller really supervises a local Python CUDA server, and the receipt reports that runtime honestly. This is a custom integration, not upstream native MiniCPM support.
