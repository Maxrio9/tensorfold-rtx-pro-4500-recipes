# Credits and notices

- [OpenBMB / MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B): base model and vision/language adaptation.
- [kuepker](https://huggingface.co/kuepker/MiniCPM-V-4.7-35B-A3B-MLX-4bit-TensorFold): measured affine4/g64 checkpoint with 8-bit routers.
- [jkuepker/TensorFold](https://github.com/jkuepker/TensorFold/tree/93fcf31c5bd861e5edb98ebf89d40e74b60b2d22): MiniCPM integration into the Python CUDA engine.
- [ashhart/TensorFold and contributors](https://github.com/ashhart/TensorFold): native benchmark CLI/protocol, Python CUDA engine, kernels, sampling, verified MTP and copying implementation.
- [Qwen and TensorFold's Qwen3.6 MTP checkpoint](https://huggingface.co/TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP): the MTP sidecar used for verified proposals.
- [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/bfcc84ab22669fd92586572d5e57ad10a5eea716): native source branch used as the benchmark-controller base.
- NVIDIA CUDA, PyTorch and Triton: inference toolchain/runtime. Existing cached container recipes came from the earlier experimental campaign.

Local work adds the supervised MiniCPM adapter/runtime facts, cache-disable integration, measurements and MTP-depth selection. It does not claim the underlying model or kernel algorithms. Source modifications are explicitly marked by patches and custom versions, prepared 17 October 2026.

Upstream engine/legacy/third-party notices and the cached target license are retained under notices. They govern their respective materials; no model is relicensed by an engine license. The MTP checkpoint's public model card declares Apache-2.0 and links Qwen3.6's terms. No weights, compiled runtime, helper binaries, private credentials or generated text/token arrays are distributed here. The unchanged public protocol module is retained only to make local receipt validation reproducible; its source URL is https://tensorfold.dev/benchmarks/protocol.js.

Qwen uses [Qwen](https://huggingface.co/Qwen), [TensorFold’s affine checkpoint](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit), [z-lab DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), and [BobClawblaw’s SM120 helper recipe](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/cada62369abb62acb761f0a3f2702d7ba217fd95). This campaign additionally forwards MiniCPM confidence correctly and evaluates proposal-policy settings. It does not alter target weights or claim these upstream algorithms.


## Qwen single-stream profiling release — 18 October 2026

- **Ash Hart / [ashhart](https://github.com/ashhart)** and TensorFold contributors: engine, existing CUDA kernels, sampler, recurrence, verified speculation and native benchmark protocol.
- **[BobClawblaw](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090)**: native SM120 recipe, CUDA helper captures and TensorFold native fork; helper recipe `cada62369abb62acb761f0a3f2702d7ba217fd95`, engine base `bfcc84ab22669fd92586572d5e57ad10a5eea716`.
- **[jkuepker](https://github.com/jkuepker/TensorFold)** and upstream CUDA contributors (including nood-co1 and Arminova): the historical Python CUDA path. The early recipe also used **turboderp's SC5 EXL3 target**, a different target/setup from the later native affine4 recipe.
- **Qwen team**: Qwen3.8-27B base model. **TensorFold contributors**: affine4/g64 target checkpoint. **[Inco AI](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)**: original DFlash2 checkpoint/method; **Z Lab / Jian Chen, Yesheng Liang and Zhijian Liu**: original DFlash work and the z-lab checkpoint mirror. Hosting a mirror does not replace original author credit.
- **NVIDIA**: CUDA compiler/runtime, Nsight Systems and Nsight Compute. **EvalPlus contributors**: HumanEval+ dataset and executable grading framework.

Local contributions are hardware-informed runtime policies, a standalone adaptation of the existing engine/kernel code, numerical checks, benchmark/quality measurements and documentation. The changes do not claim new matrix, recurrence or speculative-decoding algorithms. The patch carries upstream source context governed by the retained notices; model, drafter, helper and engine terms remain distinct. The measured tuned executable continues to report 1.0.12, so its patch and binary hashes explicitly disclose that it is modified. [Release source and evidence](qwen-tuning/).
