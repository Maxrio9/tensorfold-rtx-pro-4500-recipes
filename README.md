# TensorFold recipes for RTX PRO 4500

Reproducible CUDA recipes for running Qwen3.8-27B and MiniCPM-V-4.7-35B-A3B on a 32GB RTX PRO 4500 Blackwell. These use affine4/g64 checkpoints, SM120 kernels and verified speculative decoding, measured at stock 200W with displays active and a PCIe3 riser.

| Published recipe | Chat / code decode, tok/s | Maximum additional VRAM |
|---|---:|---:|
| [MiniCPM: MTP depth6, confidence0.7](REPRODUCE-MINICPM.md) | 361.5 / 501.3 | 20.83 GB |
| [Qwen: balanced DFlash2, verify16](REPRODUCE-QWEN.md) | 142.4 / 198.5 | 20.77 GB |

Numbers are medians of five requests in each representative receipt: one stream, 31 chat / 14 code input tokens, 256 output tokens, zero prefix reuse, BF16 KV, 2048-token serving capacity. They are client delivery rates, not engine clocks or long-context throughput. Public results and unmodified receipts are linked in [publication.json](publication.json).

MiniCPM's confidence forwarding and depth tuning improved matched chat/code medians by about 6%/8%. Depth5 is a code-focused alternative (356.2/509.1). Qwen's wider-tree experiment offered only a small matched chat gain; the original balanced recipe remains recommended. See [tuning evidence, repeatability and limitations](TUNING.md).

MiniCPM uses a custom supervised Python CUDA integration, not an upstream native MiniCPM release. These text-only tests do not qualify vision performance or broad model quality. Exact source revisions, model pins, custom versions and build hashes are in [recipes](recipes/); patches, pinned dependencies and notices accompany the reproduction guides. Fresh rebuilds need fresh receipts.

Credit belongs to the model, quantization, drafter, engine and recipe authors named in [CREDITS.md](CREDITS.md). No weights, GPU binaries, private logs, account credentials or generated text/token arrays are distributed. Supporting receipts document experiments; only one primary per model was submitted.
