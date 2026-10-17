# Build and run the pinned Qwen recipe

Use TensorFold fork commit `bfcc84ab22669fd92586572d5e57ad10a5eea716`, not a floating branch. Zig 0.17.0, CUDA 13.0.88, SM120. The native benchmark CLI and protocol remain unchanged.

```sh
git clone https://github.com/BobClawblaw/TensorFold.git tensorfold-src
git -C tensorfold-src checkout --detach bfcc84ab22669fd92586572d5e57ad10a5eea716
cd tensorfold-src
python3 -m ziglang build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dversion=1.0.12 -j2
```

Measured binary SHA256: `b09a48a2d4a3b1dc3409653ae3234deb4605c2bf9327fda2f841a324e4596c63`. A newly built executable may differ with OS/toolchain dependencies; fresh builds require their own receipt.

Fetch [BobClawblaw's helper recipe](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/cada62369abb62acb761f0a3f2702d7ba217fd95) at `cada62369abb62acb761f0a3f2702d7ba217fd95`. Preserve its SM120 helper pack; compare all 55 helper files with source/qwen-helper-kernel-manifest.json. The recipe's original recipe.toml names an older engine commit; use the newer measured engine pin above. Do not use its setup script blindly or silently change the engine revision. Download any assets under an aggregate 450 Mbit/s cap; the original campaign used 430 Mbit/s.

Prepare public HF cache snapshots for target `TensorFold/Qwen3.8-27B-MLX-4bit` at `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` and drafter `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Ensure each cache refs/main resolves to the pinned immutable snapshot, rather than current remote main. Target includes its tokenizer/config and weights; drafter precision is handled by the native loader. MLX names checkpoint layout; this run executes CUDA.

After selecting a free GPU test slot and setting HF_HUB_CACHE and the helper path:

```sh
export HF_HUB_OFFLINE=1 CUDA_MODULE_LOADING=LAZY
export TENSORFOLD_CUDA_MEMORY_LIMIT_GB=24 TF_QWEN27_GEMV_ROWS=0
export TENSORFOLD_CUDA_KERNELS=/path/to/pinned-recipe/kernels/sm120
./tensorfold-src/zig-out/native/bin/tensorfold-native benchmark \
  TensorFold/Qwen3.8-27B-MLX-4bit --backend cuda \
  --drafter z-lab/Qwen3.8-27B-DFlash2 --context 2048 --output new-benchmark.json
```

The path above is a template to replace with the downloaded helper directory. The managed CLI starts/stops its own server and disables prefix reuse. It does not restore any model. Retain native BF16 KV; no FP8 KV option is forwarded by this benchmark command. The measured container had 8 CPUs/16 GB host memory and no network. Keep telemetry with a sampled 25,000 MiB growth stop as well as the 24 GiB engine admission setting; neither is a hard allocator quota. Remain below the owner's 28 decimal GB additional-memory budget. Do not change power, displays or compute mode for an allegedly matching rerun.

## Optional wider tree

The optional source/optional-qwen-branch8.patch changes the DFlash2 tree branch count to eight and includes the managed MiniCPM bridge used by the shared experimental build. It applies to the same native base; build without overriding the patch's custom version. Its modest matched chat improvement did not establish a better balanced recipe. The published primary uses the original unmodified build above.

Credits and distinct model/engine/helper licenses are in CREDITS.md and notices/.
