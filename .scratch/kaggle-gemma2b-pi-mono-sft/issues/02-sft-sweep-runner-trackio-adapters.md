# 02: 4-Bit SFT Sweep Runner with TrackIO & Adapter Publishing (Notebook Cell 5)

**What to build:** A self-contained sweep execution engine located entirely within Cell 5 of `kaggle_gemma2b_pi_mono_sft_production.ipynb` (no external `.py` files) that trains 4-bit quantized Gemma 2 2B across 3 hyperparameter configurations for 80 steps each with completion-only loss, logging metrics to TrackIO (`sft-on-trace-v1`), pushing all intermediate LoRA adapters to Hugging Face Hub, and flushing GPU memory between runs.

**Blocked by:** 01: Setup & Trace Ingestion Pipeline (Notebook Cells 1–4)

**Status:** resolved

- [x] Implement inline response-only data collator masking prompt tokens (`<start_of_turn>model\n`) with `-100`.
- [x] Configure 3 distinct sweep configurations: `lr2e4-r16-len2k`, `lr1e4-r16-len2k`, and `lr2e4-r8-len2k`.
- [x] Load Gemma 2 2B in 4-bit precision (`unsloth/gemma-2-2b-it-bnb-4bit`) and configure LoRA on all attention and MLP projections directly in the cell.
- [x] Train each configuration for 80 steps with periodic evaluation and logging.
- [x] Extract held-out evaluation loss and training loss from trainer history.
- [x] Log metrics for each job to TrackIO project `sft-on-trace-v1`.
- [x] Push intermediate LoRA adapters to individual Hugging Face repositories (`<username>/gemma-2-2b-it-pi-mono-adapter-<job_id>`).
- [x] Perform explicit garbage collection (`gc.collect()`) and CUDA memory flushing (`torch.cuda.empty_cache()`) between sweep jobs to prevent OOM on 16GB VRAM.
- [x] Guarantee zero external `.py` dependencies so this section runs standalone within the notebook.
