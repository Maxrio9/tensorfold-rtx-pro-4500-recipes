# Further TensorFold recipe tuning on RTX PRO 4500

MiniCPM had meaningful headroom. The strongest confirmed chat setting is MTP depth 6 / confidence 0.7; depth 5 / confidence 0.7 is faster for greedy code. Qwen’s wider DFlash2 tree is a small chat-oriented option with tradeoffs. The earlier balanced Qwen recipe remains the recommendation; the new tree has not established an overall improvement. These are the best confirmed settings in this finite campaign, not a proof of a global optimum.

## Confirmed comparisons

Throughput below is client stream-delivery tok/s, concurrency 1. Each confirmed setting has three independent complete community-v1 suites, each with five measured requests per cell plus one excluded warm-up per cell. Values are medians of the three per-suite medians; they are analysis summaries, not fabricated receipts.

| Model / policy | Suites | Greedy chat | Greedy code | Sampled chat | Sampled code | Maximum additional VRAM (GB) |
|---|---:|---:|---:|---:|---:|---:|
| MiniCPM previous: depth4 / confidence0.3 | 3 | 340.71 | 463.10 | 307.47 | 415.53 | 20.855 |
| MiniCPM chat: depth6 / confidence0.7 | 3 | 361.54 | 499.02 | 320.61 | 444.50 | 20.832 |
| MiniCPM code: depth5 / confidence0.7 | 3 | 356.16 | 509.10 | 322.48 | 438.85 | 20.859 |
| Qwen control: branch4 / verify16 | 3 | 139.84 | 195.24 | 131.07 | 229.73 | 20.893 |
| Qwen chat: branch8 / verify16 | 3 | 141.98 | 195.02 | 132.44 | 227.67 | 20.904 |

MiniCPM depth6 versus matched depth4 control: **+6.11% greedy chat**, **+7.76% greedy code**. Depth5 gives +4.53% / +9.93% respectively.
Qwen branch8 versus matched branch4 control: +1.53% greedy chat, -0.11% greedy code, +1.04% sampled chat and -0.90% sampled code. The original balanced Qwen receipt remains useful: 142.39 chat / 198.52 code from the earlier campaign; cross-campaign clock, thermal and desktop variation prevent attributing that difference to policy alone.

## Actual representative receipts

Each new candidate representative is the middle greedy-chat result among its three runs. Every other cell comes from that same original receipt; no cells or samples are spliced together. MiniCPM depth6 is the new primary. Qwen retains its previously prepared balanced primary (142.39 / 198.52); the new branch8 receipt below is an optional chat-policy comparison. Supporting repeats are not queued for upload.

- **MiniCPM:** [f4705379-7d49-4da6-aacb-b17368e0a341](supporting/sweep/20261017T211126Z-mini-d6-c07-refine.json).
  - chat, T=0: 361.545 tok/s; TTFT 24.448 ms; server prefill 23.100 ms.
  - chat, T=1: 320.608 tok/s; TTFT 24.703 ms; server prefill 23.400 ms.
  - code, T=0: 501.257 tok/s; TTFT 23.864 ms; server prefill 22.800 ms.
  - code, T=1: 443.674 tok/s; TTFT 24.532 ms; server prefill 23.100 ms.
- **Qwen optional branch8:** [a83e8229-5296-42d1-80f4-6cfa559a00e7](supporting/sweep/20261017T211646Z-qwen-b8-confirm-b.json).
  - chat, T=0: 141.981 tok/s; TTFT 84.329 ms; server prefill 83.254 ms.
  - chat, T=1: 132.658 tok/s; TTFT 83.967 ms; server prefill 82.941 ms.
  - code, T=0: 195.023 tok/s; TTFT 82.011 ms; server prefill 81.024 ms.
  - code, T=1: 229.438 tok/s; TTFT 82.112 ms; server prefill 81.154 ms.

Inputs are actually **14 code / 31 chat tokens**, 256 output tokens, context capacity 2048, BF16 KV, **all measured cached_tokens=0**. These are short-input decode results, not 16k/64k or concurrency claims. Prefill figures at these tiny inputs mostly describe latency; do not extrapolate them into long-prompt processing speed. Compilation/model loading are outside warmed measured cells.

## What changed and why

- MiniCPM’s family wrapper accepted the CLI confidence option but did not forward it to the actual CUDA MTP engine. The new explicit argument forwarding fixes that. Increasing confidence stops extending lower-confidence draft chains; this makes larger maximum depths practical without paying for as many doomed proposals. The target verifier still determines emitted tokens. Copy proposals remain enabled as upstream. No target/drafter weights, quantization, activation arithmetic or CUDA kernels were changed.
- A separate greedy-chat probe supports the MiniCPM mechanism: engine-reported proposals fell from 349 to 248, accepted proposals from 155 to 149, and rounds rose from 100 to 106. Less speculative work can still win despite more rounds. For greedy code, rounds fell 70 to 64 and proposals 278 to 261. Counts include upstream copy behavior; timings from those numerical probes are not leaderboard scores.
- The small Qwen median gain is comparable to variation in its control runs and does not establish a strong overall replacement. Qwen changes only the host proposal-tree branch count from four to eight, retaining the same 16-row target verification and SM120 captured helpers. More branches provide different candidate coverage; the finite row budget can benefit one workload and cost another. Branch two lost substantial chat throughput and was rejected.
- MiniCPM .9 confidence regressed; depth7/.7 also regressed. Nearby .6 and .8 settings did not demonstrate a better combined result. Depth5 versus depth6 is a measured code/chat tradeoff, not a prompt-dependent switch. No fixture recognition or selective benchmark behavior was added.

## Protocol and correctness

All standard receipts pass the captured official TensorFold protocol validator. Suite digest, prompts, seeds, temperatures, top-k/p, EOS behavior, timing and hardware/model provenance remain intact. Public target and head revisions are pinned. The native fixture, stream timing, hardware, validation, model and receipt modules are byte-identical to the pinned source base; unchanged-protocol-files.json records hashes. The bridge changes supervised server launch and truthful runtime facts only.
Native host policy tests passed, as did 22 Python family tests (one MLX-only test skipped) and CPU constructor-forwarding checks. All 60 full draft/plain comparisons passed across five tested configurations. These checks include benchmark and unseen prompts, greedy and sampled generation. Full MiniCPM token arrays are compared locally; Qwen compares full server-provided SHA256 values, not just the truncated receipt fingerprint. All completed standard suites also retain identical measured token fingerprints versus their same-model control. These finite checks do not constitute a broad model-quality or mathematical exactness proof. The leaderboard and these correctness checks are text-only; MiniCPM vision inference was not qualified in this campaign.
MiniCPM is a **custom managed Python CUDA integration** based on jkuepker’s fork. Its receipt truthfully reports Python 3.12.3 and the custom engine version; it is not represented as upstream native MiniCPM support. Source builds are documented by the [official benchmark guide](https://github.com/ashhart/TensorFold/blob/main/docs/community-benchmarks.md); candidate schema validation is not a guarantee of upstream endorsement or future live receiver acceptance. The two primary receipts have since been approved and read back unchanged from TensorFold; see publication.json. No supporting repeats or optional variants were uploaded.

## Limits and remaining work

Stock 200W, active desktop/displays, Compute Mode Default, ECC off, PCIe3 riser, 32GB DDR4 host. No display, power, compute-mode or clocks were changed. The watchdog sampled GPU growth every 0.2s and aborted at 24,000MiB additional usage, below the authorized 28 decimal GB cap. Sampled peaks can miss short transients. Final timed suites ran without other benchmark/build containers; desktop activity can still affect results.
Further worthwhile work would be held-out long-form workload performance, compute/display mode comparisons with explicit availability, and kernel/layout work with fresh numerical validation. Those are separate campaigns. No need to repeat weak EXL3/NVFP4 Python recipes or the already completed Qwen GEMV/context/compiler sweeps merely to claim more experiments.

This campaign preserved **23 complete suites / 460 measured requests**, plus excluded warmups and separate numerical probes. Every candidate, including rejected configurations, remains in summary.json and groups.json. Pilot-only configurations have one suite and are provisional; selected/control configurations have three. No model is to be restored after completion. This public snapshot excludes local logs, runtime assets, host paths and credentials. Detailed research is maintained separately.

## All tested settings

| Model | Depth | Confidence | Qwen branch | Suites | Chat T0 | Code T0 | Chat T1 | Code T1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini | 4 | 0.3 | — | 3 | 340.71 | 463.10 | 307.47 | 415.53 |
| mini | 4 | 0.5 | — | 1 | 348.86 | 474.36 | 316.08 | 427.63 |
| mini | 4 | 0.7 | — | 1 | 352.01 | 485.57 | 320.34 | 427.85 |
| qwen | — | — | 4 | 3 | 139.84 | 195.24 | 131.07 | 229.73 |
| qwen | — | — | 2 | 1 | 127.66 | 195.05 | 123.61 | 227.15 |
| qwen | — | — | 8 | 3 | 141.98 | 195.02 | 132.44 | 227.67 |
| mini | 4 | 0.9 | — | 1 | 342.93 | 459.68 | 312.42 | 417.92 |
| mini | 3 | 0.7 | — | 1 | 340.46 | 458.34 | 312.10 | 394.38 |
| mini | 5 | 0.7 | — | 3 | 356.16 | 509.10 | 322.48 | 438.85 |
| mini | 6 | 0.7 | — | 3 | 361.54 | 499.02 | 320.61 | 444.50 |
| mini | 7 | 0.7 | — | 1 | 359.32 | 498.55 | 317.35 | 431.85 |
| mini | 5 | 0.6 | — | 1 | 359.33 | 497.65 | 318.79 | 436.79 |
| mini | 5 | 0.8 | — | 1 | 358.57 | 494.94 | 323.38 | 435.23 |
