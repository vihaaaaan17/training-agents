# 04: Inspect AI Coding Evals & Model Card Generator (Notebook Cells 7–8)

**What to build:** An automated benchmarking and documentation pipeline that evaluates the final merged model on HumanEval and MBPP coding benchmarks via Inspect AI within the local sandbox, parses pass@1 accuracy scores, and compiles and uploads a comprehensive model card `README.md` to the final repository, implemented entirely within Cells 7–8 of `kaggle_gemma2b_pi_mono_sft_production.ipynb` (no external `.py` files).

**Blocked by:** 03: Best Run Selection & 16-Bit Model Merge (Notebook Cell 6)

**Status:** resolved

- [x] Execute Inspect AI evaluation for `inspect_evals/humaneval` against the merged model in Cell 7 using the local sandbox with sample-budget limits.
- [x] Execute Inspect AI evaluation for `inspect_evals/mbpp` against the merged model in Cell 7 using the local sandbox with sample-budget limits.
- [x] Parse JSON execution logs generated in `./inspect_logs` to extract pass@1 accuracy metrics.
- [x] Format a complete Markdown `README.md` in Cell 8 containing:
  - Benchmark evaluation scores table (HumanEval & MBPP).
  - Hyperparameter sweep comparison table indicating the winning run.
  - Links to TrackIO dashboard, dataset source, and final repository.
  - Documented evaluation limitations and training scope.
  - Verification & reproduction details.
- [x] Upload the generated `README.md` directly to the final Hugging Face model repository.
- [x] Guarantee zero external `.py` dependencies so this section runs standalone within the notebook.
