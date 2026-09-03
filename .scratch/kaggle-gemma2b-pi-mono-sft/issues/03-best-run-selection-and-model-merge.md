# 03: Best Run Selection & 16-Bit Model Merge (Notebook Cell 6)

**What to build:** Automated checkpoint ranking based on minimum held-out evaluation loss, full 16-bit weight merging of the winning adapter into the base model, and publishing the standalone merged model to the final Hugging Face repository, implemented entirely within Cell 6 of `kaggle_gemma2b_pi_mono_sft_production.ipynb` (no external `.py` files).

**Blocked by:** 02: 4-Bit SFT Sweep Runner with TrackIO & Adapter Publishing (Notebook Cell 5)

**Status:** resolved

- [x] Inspect sweep results programmatically to identify the winning run by lowest held-out evaluation loss.
- [x] Surface winning job ID, best evaluation loss, and associated hyperparameters.
- [x] Load the winning adapter checkpoint into memory.
- [x] Merge the adapter weights into base model weights in full 16-bit precision using `save_pretrained_merged`.
- [x] Push the merged standalone model and tokenizer to the final Hugging Face repository (`<username>/gemma-2-2b-it-pi-mono-sft`).
- [x] Flush GPU VRAM and run garbage collection following the merge operation.
- [x] Guarantee zero external `.py` dependencies so this section runs standalone within the notebook.
