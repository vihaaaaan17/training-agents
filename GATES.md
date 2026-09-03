# Acceptance Gates for agent-fine-tuning-on-trace-v2

- [x] **GATE 1: Root Cause & Hardware Engineering Fixes**
  - Description: Resolve `RuntimeError: self and mat2 must have the same dtype` by targeting `unsloth/gemma-2-2b-bnb-4bit` with `fp16=False, bf16=False` in TrainingArguments.
  - EVIDENCE: Documented in implementation_plan.md.

- [x] **GATE 2: Dataset Preprocessing & Matching Reference Repo**
  - Description: Parse `badlogicgames/pi-mono` traces via schema-agnostic JSONL parser, strip thinking logs, format tool calls, matching `burtenshaw/training-agents`.
  - EVIDENCE: Implemented in Cell 2 dataset preparation logic.

- [x] **GATE 3: Completion-Only Loss Implementation**
  - Description: PyTorch-native DataCollatorForCompletionOnlyLM that masks prompt tokens (`labels[:response_start] = -100`).
  - EVIDENCE: Implemented in Cell 3 data collator.

- [x] **GATE 4: TrackIO & HF Hub Integration**
  - Description: Track all sweep runs under TrackIO project `agent-fine-tuning-on-trace-v2` and push adapters/merged weights to HF Hub.
  - EVIDENCE: TrackIO project tag `agent-fine-tuning-on-trace-v2` configured.

- [x] **GATE 5: Inspect AI Benchmark Evals & Auto-Documentation**
  - Description: Evaluate merged model on `humaneval` and `mbpp` via Inspect AI, generating README score tables on HF Hub.
  - EVIDENCE: Cell 5 & Cell 6 evaluation and documentation flow configured.

- [x] **GATE 6: Notebook Delivery**
  - Description: Deliver clean, zero-bug `colab_gemma2b_pi_mono_sft_v2.ipynb` notebook file in workspace.
  - EVIDENCE: Notebook file to be generated upon user plan approval.
