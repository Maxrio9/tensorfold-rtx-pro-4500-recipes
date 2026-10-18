# Reproduce the tuned single-stream Qwen recipe

This page builds the exact standalone engine patch and freezes the selected runtime policy. It does not download weights or start inference until you explicitly run the serving/benchmark commands.

## Build the engine

Requirements: Linux, NVIDIA driver compatible with CUDA 13, CUDA compiler **13.0.88**, **Zig 0.17.0**, Git, Python 3 and the native build dependencies. [Dockerfile.builder](source/Dockerfile.builder) records the measured builder; its pinned vLLM image is a compiler environment, not the inference engine. Rebuilding that image needs network access for apt/pip; inference can run offline.

From this repository root:

```sh
recipe_root="$PWD/qwen-tuning"
git clone https://github.com/BobClawblaw/TensorFold.git tensorfold-tuned-src
git -C tensorfold-tuned-src checkout --detach bfcc84ab22669fd92586572d5e57ad10a5eea716
git -C tensorfold-tuned-src apply --check "$recipe_root/source/tunable-v13.patch"
git -C tensorfold-tuned-src apply "$recipe_root/source/tunable-v13.patch"
cd tensorfold-tuned-src
python3 -m ziglang build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dversion=1.0.12 -j2
cd ..
```

Install `ziglang==0.17.0` into your chosen Python build environment if needed. Alternatively build [Dockerfile.builder](source/Dockerfile.builder), mount the patched checkout at `/src`, and run the same build command there. No GPU is needed for compilation. Native executable: `tensorfold-tuned-src/zig-out/native/bin/tensorfold-native`.

The measured tuned binary hash is `5ceb4b081f95a243c55fdf4870da677f8716253252c6ef463641f6f6e201a5da`; patch hash `92691453fe7d966b50ac3e9da14977e0cad743f01f5c9a638d953390c2431fb2`. Fresh toolchains may produce another binary hash; generate fresh results and record that hash. The native runtime reports version 1.0.12 even with this patch; disclose the modification alongside any receipt.

## Obtain pinned helpers and checkpoints

Use [BobClawblaw's SM120 helper recipe](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/cada62369abb62acb761f0a3f2702d7ba217fd95), commit `cada62369abb62acb761f0a3f2702d7ba217fd95`, with its original `kernels/sm120` pack. Verify it against [the existing helper manifest](../source/qwen-helper-kernel-manifest.json). Do not run its setup script blindly: the engine base for these results is **bfcc84ab**, plus our patch. The single-stream profile does not need the extra attention captures made for experimental adaptive batching.

Download complete immutable snapshots (config, tokenizer and weights):

| Asset | Public repository | Revision |
|---|---|---|
| Target | [TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit) | `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` |
| Drafter | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

The drafter mirrors the Inco AI release; credit the original DFlash2 authors. MLX names the checkpoint layout; this recipe executes CUDA. Keep aggregate model downloads at or below **450 Mbit/s**, including parallel connections. No weights or helper binaries are redistributed here.

## Start the selected service

Replace the asset paths below with your local pinned snapshots. `run.py` reads every selected environment setting from [recipe.json](recipe.json); its `profile.flags` array is historical launcher metadata, not native CLI flags.

```sh
python3 qwen-tuning/run.py serve --allow-desktop \
  --engine ./tensorfold-tuned-src/zig-out/native/bin/tensorfold-native \
  --helpers /path/to/pinned-recipe/kernels/sm120 \
  --target /path/to/pinned-target-snapshot \
  --drafter /path/to/pinned-drafter-snapshot
```

The service binds localhost port 19090, uses one lane, 2048-token serving capacity, BF16 KV and disabled prefix retention. The wrapper checks for other compute processes before starting, samples additional whole-GPU memory every 0.2 seconds, stops its owned process group at 26000 MiB growth and cleans up on exit. It never restores a model. Sampling is not a hard allocation guarantee; remain below **28 decimal GB additional GPU memory**. Run without other model/OCR compute jobs. `--allow-desktop` permits the Chromium/Chrome, voxtype OSD and Nautilus render contexts present in this desktop setup; omit it for a strict no-other-compute-process check. It does not change displays or Compute Mode. Keep power, clocks, display and Compute Mode conditions consistent with the comparison you report.

## Reproduce the graphic's 512-output workload

Install the Hugging Face `tokenizers` Python package in your client environment. In another terminal, with the service already running:

```sh
python3 qwen-tuning/benchmark-service.py \
  --tokenizer /path/to/pinned-target-snapshot/tokenizer.json \
  --out new-tuned-service-run
```

This portable adapter retains the original prompt, streaming/timing and burst functions verbatim. It replaces the machine-specific campaign controller. It performs one excluded 512-token warmup per task and five measured runs each for prose/code, with fresh deterministic prompt tags, T0.6/top_p0.95/top_k20/min_p0/seed42, thinking off and fixed 512 output tokens. API output counts and actual cached counts must pass checks. Mean client decode excludes the retokenized complete first delivered text block; whole-request aggregate includes prefill and drain. It saves raw streams **locally** for audit, never publishes them, and does not load/unload a model.

For a matching baseline, build the clean engine at the same base commit and use the original helper pack with the original serving configuration, followed by the same client script. The [baseline scalar confirmation](evidence/baseline-service-samples.json) records the original measurements. Do not apply the tuned environment to an unpatched baseline and assume the settings took effect.

## Optional standard community-v1 receipt

First stop your owned service so the GPU is free. Prepare a normal HF hub cache whose target/drafter `refs/main` resolve to the exact revisions above. Then:

```sh
python3 qwen-tuning/run.py benchmark --allow-desktop \
  --engine ./tensorfold-tuned-src/zig-out/native/bin/tensorfold-native \
  --helpers /path/to/pinned-recipe/kernels/sm120 \
  --hf-cache /path/to/pinned-huggingface-hub-cache \
  --output new-tuned-benchmark.json
node validation/validate_receipt.mjs new-tuned-benchmark.json
```

The native CLI owns its server, uses the original 256-output fixture and defaults, and writes a local receipt. The wrapper adds a provenance sidecar with binary/recipe/patch hashes. No upload occurs. A new build, new score or source changes need their own disclosed receipt; protocol validation is not leaderboard approval.

## Quality and numerical validation

The speed protocol does not reproduce the quality score: quality uses thinking on, T1/top_p0.95/top_k20/min_p0/seed42, `reasoning_effort: low`, 8192-token capacity, 4096-token reply budget and natural EOS. All 164 coding tasks plus 42 application fixtures get one sample, no retries or repair. Coding prompts prepend: `Implement the following Python function. Return the complete Python code including the shown function signature and any needed imports. Do not include tests or explanations.`, then the public task prompt. Do not send reference answers or hidden tests to the model.

[Dataset/EvalPlus pins](evidence/quality-source.json), [strict grader](validation/grade.py), [application fixtures/scorer](validation/application-fixtures.py), per-task grades and hashes are included. Grade untrusted code only in a CPU-only isolated sandbox with no GPU, network, user files, credentials or Docker socket. Generated responses remain local and are intentionally absent from this release.

[Split-K oracle](validation/splitk_oracle.py) and [pipeline oracle](validation/qmm_oracle_v3.py) retain the recorded container-oriented diagnostic scripts; they are not portable launchers. Their `/src`, `/models`, `/study` and `/results` paths are container mount placeholders, and they require PyTorch plus CUDA Python bindings and rebuilt kernel fatbins. Read the source and predeclared error bounds before adapting them. These diagnostic timings are not model scores.
