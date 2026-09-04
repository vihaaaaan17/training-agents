# 02: Kaggle-Native SFT Sweep Runner & Winning Model Merge

**What to build:** An execution pipeline in the training notebook that executes a 3-job hyperparameter sweep on Gemma 2 2B with completion-only loss under Kaggle free-tier constraints (Tesla T4 16GB VRAM), logs run metrics to TrackIO, selects the best adapter via lowest held-out evaluation loss, merges the winning adapter into 16-bit weights, and pushes the final model to Hugging Face Hub as `orangefabercastell/gemma-2-2b-it-pi-mono-sft-v2`.

**Blocked by:** 01: Canonical Inlined Tool-Call Trace Pipeline & 70/30 Hybrid Blend

**Status:** ready-for-agent

- [ ] Configure 4-bit QLoRA training with `unsloth/gemma-2-2b-it-bnb-4bit` or `google/gemma-2-2b-it` using `bfloat16` compute.
- [ ] Run 3-job sweep exploring learning rates (`1e-4`, `2e-4`) and LoRA configurations (`r=16, alpha=32` vs `r=8, alpha=16`) with `completion_only_loss=True`.
- [ ] Log real-time loss, token accuracy, and GPU memory metrics to TrackIO.
- [ ] Automatically select the winning run based on minimum held-out evaluation loss.
- [ ] Merge the winning LoRA adapter with the base model in 16-bit float precision on CPU to prevent CUDA OOM.
- [ ] Push the winning adapter and merged 16-bit model to Hugging Face Hub under `orangefabercastell/gemma-2-2b-it-pi-mono-sft-v2`.
- [ ] Run a qualitative sanity check cell verifying that the model outputs valid `<tool_call>` blocks on test prompts.
