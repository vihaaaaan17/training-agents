# 0001: Kaggle-Native SFT Sweep Pipeline

We run the SFT hyperparameter sweep directly on Kaggle's free GPU (Tesla T4 16GB) using 4-bit quantized Gemma 2 2B (`unsloth/gemma-2-2b-it-bnb-4bit`) and completion-only loss rather than dispatching remote Hugging Face Jobs. This ensures the workflow is 100% cost-free within Kaggle's 30h/week GPU quota, avoids paid cloud compute credits, eliminates remote CLI orchestration failure modes, and allows Inspect AI evaluations to run directly in Kaggle's local container sandbox.
