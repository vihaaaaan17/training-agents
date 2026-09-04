# 04: Side-by-Side Empirical Evaluation & Verification Scorecard

**What to build:** An evaluation execution pipeline in `kaggle-eval-trace-benchmarks.ipynb` that runs base Gemma 2 2B vs. the newly trained SFT v2 model (`orangefabercastell/gemma-2-2b-it-pi-mono-sft-v2`), guarantees zero VRAM leaks via protected model teardown, computes metrics (Tool Invocation Rate, Tool Match Accuracy, Valid Arguments Rate, Latency), and persists verifiable scorecard artifacts (`benchmark_results_raw.json` and `benchmark_scorecard_summary.csv`).

**Blocked by:** 02: Kaggle-Native SFT Sweep Runner & Winning Model Merge, 03: Parameter-Complete Benchmark Ingestion for BFCL & ToolBench

**Status:** ready-for-agent

- [ ] Execute evaluation pipeline for Base Model (`google/gemma-2-2b-it` or `unsloth/gemma-2-2b-it`) and verify clean VRAM teardown.
- [ ] Execute evaluation pipeline for SFT v2 Model (`orangefabercastell/gemma-2-2b-it-pi-mono-sft-v2`) and verify clean VRAM teardown.
- [ ] Verify empirical deltas: confirm SFT v2 scores significant positive gains (> +40%) on Tool Invocation Rate and Tool Selection Accuracy over Base.
- [ ] Render styled comparison table and persist all 400 execution traces in `benchmark_results_raw.json` and summary table in `benchmark_scorecard_summary.csv`.
- [ ] Update root `README.md` with verified benchmark tables and new model card links.
