# 01: Setup & Trace Ingestion Pipeline (Notebook Cells 1–4)

**What to build:** An end-to-end data pipeline implemented strictly inside Cells 1–4 of `kaggle_gemma2b_pi_mono_sft_production.ipynb` (no external `.py` files) that installs dependencies, authenticates via Kaggle secrets, parses raw agent execution traces from `badlogicgames/pi-mono` into prompt/completion pairs with inline canonical tool schemas, and creates validated train and held-out evaluation splits constrained to 2,048 tokens.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Core dependencies installed in Cell 1 (`unsloth`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `trackio`, `inspect-ai`, `datasets`, `huggingface_hub`).
- [x] Authenticate Hugging Face write token via Kaggle Secrets (`HF_TOKEN`) in Cell 2 with fallback to interactive input.
- [x] Implement inline canonical tool schemas (`bash`, `read`, `edit`, `write`, `grep`, `find`, `ls`, `todo`) and tool call conversion in Cell 3.
- [x] Download raw `*.jsonl` files from `badlogicgames/pi-mono` and parse messages directly in Cell 4 without PyArrow casting errors.
- [x] Strip internal thinking/reasoning parts and omit image tokens (`[image omitted]`) directly within the cell parser.
- [x] Implement context trimming with user message anchoring and character-based compaction inline.
- [x] Apply Gemma chat template to format visible turns into `prompt` and `completion` pairs.
- [x] Filter out overlength sequences exceeding 2,048 tokens.
- [x] Create deterministic train and held-out evaluation dataset splits with fixed seed.
- [x] Guarantee zero external `.py` dependencies so this section runs standalone within the notebook.
