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
