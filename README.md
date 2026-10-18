# TensorFold recipes for RTX PRO 4500

Reproducible CUDA recipes for running Qwen3.8-27B and MiniCPM-V-4.7-35B-A3B on a 32GB RTX PRO 4500 Blackwell. These use affine4/g64 checkpoints, SM120 kernels and verified speculative decoding, measured at stock 200W on a system with a PCIe3 riser. Display conditions are recorded per study.

For the newer **Qwen single-stream tuning**, see [results, engine patch and reproduction guide](qwen-tuning/). Fresh matched 512-output tests improved prose/code decode from **140.7/187.2 to 156.7/215.4 tok/s**, with separate thinking-on quality checks of **153/164 HumanEval+ and 42/42 application tasks**. These supporting measurements are separate from the earlier published leaderboard receipts below.

| Published recipe | Chat / code decode, tok/s | Maximum additional VRAM |
|---|---:|---:|
| [MiniCPM: MTP depth6, confidence0.7](REPRODUCE-MINICPM.md) | 361.5 / 501.3 | 20.83 GB |
| [Qwen: balanced DFlash2, verify16](REPRODUCE-QWEN.md) | 142.4 / 198.5 | 20.77 GB |

Numbers are medians of five requests in each representative receipt: one stream, 31 chat / 14 code input tokens, 256 output tokens, zero prefix reuse, BF16 KV, 2048-token serving capacity. They are client delivery rates, not engine clocks or long-context throughput. Public results and unmodified receipts are linked in [publication.json](publication.json).

MiniCPM's confidence forwarding and depth tuning improved matched chat/code medians by about 6%/8%. Depth5 is a code-focused alternative (356.2/509.1). Qwen's earlier wider-tree experiment offered only a small matched chat gain; the original balanced recipe remains the published leaderboard baseline. The newer single-stream engine adaptation is documented separately above. See [tuning evidence, repeatability and limitations](TUNING.md).

MiniCPM uses a custom supervised Python CUDA integration, not an upstream native MiniCPM release. These text-only tests do not qualify vision performance. The newer Qwen study includes finite coding/application checks, with scope and limitations recorded on its page. Exact source revisions, model pins, custom versions and build hashes are in [recipes](recipes/); patches, pinned dependencies and notices accompany the reproduction guides. Fresh rebuilds need fresh receipts.

Credit belongs to the model, quantization, drafter, engine and recipe authors named in [CREDITS.md](CREDITS.md). No weights, GPU binaries, private logs, account credentials or generated text/token arrays are distributed. Supporting receipts document experiments; only one primary per model was submitted. The newer tuned Qwen receipts have not been submitted.
