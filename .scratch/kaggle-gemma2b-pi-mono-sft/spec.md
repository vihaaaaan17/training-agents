# Spec: Kaggle-Native SFT Sweep for Gemma 2 2B on Pi-Mono Traces

Status: ready-for-agent

## Problem Statement
Fine-tuning lightweight agent models on execution traces typically requires expensive cloud infrastructure or complex multi-machine orchestration that is difficult to reproduce and prone to out-of-memory errors on free-tier accelerators. Furthermore, multi-file Python repositories (`src/*.py`, helper modules) create friction in cloud notebook environments like Kaggle where users must upload, mount, or install custom packages. Machine learning engineers need a **100% self-contained, single `.ipynb` notebook** that can execute the entire end-to-end pipeline (data parsing, 3-job sweep, TrackIO logging, weight merging, Inspect AI coding evals, and README publishing) natively under Kaggle's free GPU limits (Tesla T4 16GB VRAM) with zero external Python file dependencies.

## Solution
A single, self-contained, production-grade Jupyter notebook (`kaggle_gemma2b_pi_mono_sft_production.ipynb`) that requires **no external `.py` files** and contains all logic inline:
1. **Cells 1–4 (Setup & Ingestion)**: Replicates the exact trace ingestion and tool schema formatting from `training-agents` to convert raw `badlogicgames/pi-mono` traces into prompt/completion examples with context trimming.
2. **Cell 5 (Sweep Engine)**: Executes a 3-configuration hyperparameter sweep (80 steps each) using 4-bit quantized Gemma 2 2B (`unsloth/gemma-2-2b-it-bnb-4bit`) and completion-only loss. Logs runs to TrackIO project `sft-on-trace-v1` and pushes all LoRA adapters to Hugging Face Hub.
3. **Cell 6 (Model Merge)**: Automatically selects the winning checkpoint based on held-out evaluation loss, merges the adapter into full 16-bit weights, and uploads the standalone model to the target repository.
4. **Cells 7–8 (Evals & Card)**: Runs Inspect AI coding benchmarks (`humaneval` and `mbpp`) locally in the Kaggle sandbox, parses pass@1 accuracy, and publishes a complete model card `README.md` to Hugging Face Hub.

## User Stories

1. As an ML engineer, I want the entire workflow contained in a **single `.ipynb` file without external `.py` files**, so that I can upload one notebook to Kaggle and run it immediately without repository cloning or custom package installation.
2. As an ML engineer, I want the notebook to authenticate automatically using Kaggle Secrets (`HF_TOKEN`), so that my credentials remain private while granting Hub write access.
3. As an ML engineer, I want the dataset downloader to parse raw `*.jsonl` files directly from `badlogicgames/pi-mono`, so that I bypass PyArrow dataset-server schema casting failures.
4. As an ML engineer, I want assistant turns to be formatted with inline standard tool definitions (`bash`, `read`, `edit`, `write`, `grep`, `find`, `ls`, `todo`), so that the model learns valid agentic tool call invocations.
5. As an ML engineer, I want internal thinking and reasoning parts stripped from the dataset, so that fine-tuning only trains on observable output actions.
6. As an ML engineer, I want long multi-turn context windows trimmed with user message anchoring, so that the prompt remains coherent and within token limits.
7. As an ML engineer, I want overlength examples filtered out before batching, so that sequences strictly respect the 2,048 token boundary.
8. As an ML engineer, I want completion-only loss applied to assistant turns via an inline data collator, so that user inputs and environment tool results are masked out (`-100`).
9. As an ML engineer, I want the sweep to evaluate 3 discrete parameter combinations (`lr=2e-4, r=16`, `lr=1e-4, r=16`, `lr=2e-4, r=8`) for 80 steps each, so that I can identify optimal learning dynamics within Kaggle's execution window.
10. As an ML engineer, I want GPU cache cleared (`torch.cuda.empty_cache()` and `gc.collect()`) between sweep jobs, so that successive runs do not accumulate VRAM and crash with OOM errors.
11. As an ML engineer, I want each sweep job to publish its LoRA adapter weights to a distinct Hugging Face Hub repository, so that intermediate checkpoints are preserved.
12. As an ML engineer, I want all sweep runs logged to TrackIO under project `sft-on-trace-v1`, so that I have centralized experiment tracking and loss curves.
13. As an ML engineer, I want the notebook to select the best checkpoint using held-out evaluation loss, so that model selection is driven objectively by generalization performance.
14. As an ML engineer, I want the winning adapter merged into 16-bit precision and pushed to a final repository, so that downstream users can run inference without PEFT dependencies.
15. As an ML engineer, I want the final model evaluated on `humaneval` and `mbpp` using Inspect AI, so that I have standardized coding benchmark pass@1 metrics.
16. As an ML engineer, I want Inspect AI to run inside Kaggle's local container sandbox, so that evaluations execute without requiring Docker daemons or vLLM server overhead.
17. As an ML engineer, I want an automated `README.md` uploaded to the final repository containing benchmark scores, sweep job IDs, and documented limits, so that the model card is fully reproducible.

## Implementation Decisions

- **Single Notebook Architecture**: Strict constraint that all functionality lives entirely inside `kaggle_gemma2b_pi_mono_sft_production.ipynb`. No local `.py` imports or multi-file dependencies.
- **Domain Model Compliance**: Adheres to terms defined in `CONTEXT.md` (*Trace*, *Assistant Example*, *Completion-Only Loss*, *Sweep Job*, *Held-Out Eval Loss*).
- **Architecture**: Enforces ADR 0001 (Kaggle-native execution on Tesla T4 16GB GPU with 4-bit QLoRA instead of remote HF Jobs).
- **Base Checkpoint**: `unsloth/gemma-2-2b-it-bnb-4bit` for zero-overhead loading in ~3.8 GB VRAM.
- **Collator Strategy**: Implements response-only masking targeting `<start_of_turn>model\n`.
- **Target Modules**: LoRA applied across all attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Evaluation Sandbox**: Uses Inspect AI with `--sandbox local` and `--limit 10` sampling for free-tier budget compliance.

## Testing Decisions

- **Notebook Validation**: The single `.ipynb` file must parse cleanly against Jupyter nbformat v4 and execute sequentially from top to bottom with zero missing import errors.
- **Seam Testing**: Test external behavior at the three defined seams (data conversion, sweep execution, evaluation/publishing).
- **Failure Resilience**: Test that network timeouts or TrackIO logging errors are caught gracefully without terminating the sweep loop.

## Out of Scope

- Splitting code into multiple `.py` modules or local packages.
- Remote Hugging Face Jobs provisioning (superseded by ADR 0001).
- Multi-node distributed DDP/FSDP training.
- Docker-based evaluation harnesses (such as SWE-bench or BigCodeBench).

## Further Notes
The entire workflow is packaged into `kaggle_gemma2b_pi_mono_sft_production.ipynb`.
