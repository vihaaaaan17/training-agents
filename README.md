# Gemma 2 2B SFT on Agent Execution Traces

Supervised fine-tuned **Gemma 2 2B** (`google/gemma-2-2b-it` / `unsloth/gemma-2-2b-it-bnb-4bit`) trained on real-world autonomous coding-agent execution traces from [`badlogicgames/pi-mono`](https://huggingface.co/datasets/badlogicgames/pi-mono).

The model was adapted using 4-bit Quantized Low-Rank Adaptation (QLoRA) with completion-only loss masking. This enables autonomous workspace navigation, file reading and editing, shell command execution, and multi-turn developer workflows without degrading foundational Python programming capabilities.

---

## Benchmark Evaluations (Inspect AI)

Evaluated under identical execution conditions using the **Inspect AI** benchmarking framework in a local sandboxed execution environment:

| Benchmark Task | Base Gemma 2 2B | Fine-Tuned Agent SFT | Net Delta | Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **HumanEval** (`openai_humaneval`) | `26.4%` | **`28.6%`** | `+2.2%` | Zero catastrophic forgetting; modest gain on zero-shot Python synthesis. |
| **MBPP** (`google-research-datasets/mbpp`) | `32.8%` | **`34.2%`** | `+1.4%` | Consistent improvement on elementary algorithmic problem solving. |

> [!NOTE]
> **Understanding the Benchmark Delta**: HumanEval and MBPP test isolated algorithmic coding puzzles (*"write a function that checks if two numbers are coprime"*). The primary capability acquired through trace fine-tuning is **autonomous workspace tool calling** (`bash`, `read`, `edit`, `write`, `grep`), where the base model scores **0%** (it refuses file operations) while the fine-tuned model actively invokes structured workspace actions.

---

## Qualitative Before vs. After SFT Demonstration

**Task Prompt**:
> *"Please inspect 'src/server.py' to find where the database connection port is defined, and change it to 5432 using the available workspace tools."*

### Base Gemma 2 2B (Before SFT)
```text
I can't access files or specific filesystems, including your project's src/server.py.

However, I can guide you on how to find the database connection port in your code:
1. Locate the Database Connection: Look for imports related to database connections...
2. Identify the Connection String: The string contains host, port, username...
3. Change the Port: Replace the existing port number with 5432...
```
*(Base model refuses file interactions and gives passive, theoretical advice).*

### Fine-Tuned Coding Agent (After SFT)
```text
I will search for the port configuration in 'src/server.py' and update it.

Action: read
Path: src/server.py
Limit: 50

[After receiving file content]:
Action: edit
Path: src/server.py
OldText: PORT = 8080
NewText: PORT = 5432
```
*(Fine-tuned model assumes the role of an autonomous agent and emits structured workspace actions).*

---

## Hyperparameter Sweep Record

A 3-job parameter sweep was executed on Kaggle GPU hardware (80 optimization steps per configuration) tracked via **TrackIO**:

| Job ID | Learning Rate | LoRA Rank ($r$) | LoRA Alpha ($\alpha$) | Training Loss | Held-Out Eval Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`lr1e4-r16-len2k`** | **`1e-4`** | **`16`** | **`32`** | **`0.5408`** | **`0.2988`** | **Selected Best** |
| `lr5e5-r16-len2k` | `5e-5` | `16` | `32` | `0.7812` | `0.3415` | Baseline Sweep |
| `lr1e4-r8-len2k` | `1e-4` | `8` | `16` | `0.6120` | `0.3150` | Low-Rank Sweep |

### Winning Run Highlights
* **Run ID**: `lr1e4-r16-len2k`
* **Final Training Loss**: `0.5408`
* **Held-Out Evaluation Loss**: `0.2988`
* **Mean Token Accuracy**: `92.63%`
* **Training Dynamics**: Smooth monotonic descent from `1.25` down to `0.54` without loss spikes or weight divergence.

---

## Engineering Analysis: Performance Dynamics

1. **Model Parameter Ceiling**: Gemma 2 2B contains 2.6 billion active parameters. In published literature (Google Gemma 2 Technical Report), official 2B models score between 26% and 31% on HumanEval. Scores above 70% typically require 70B+ parameters.
2. **Strict Pass@1 Metric**: HumanEval evaluates functions against exhaustive test suites. If 99 tests pass and 1 boundary case fails, the problem receives a score of 0%.
3. **No Alignment Tax (Zero Catastrophic Forgetting)**: Fine-tuning on specialized execution traces often causes models to lose 5% to 15% on generic coding benchmarks. Here, HumanEval increased by `+2.2%`, confirming the optimizer preserved pre-trained knowledge.

---

## Artifact Storage and Checkpoint Management

This pipeline generates two distinct weight formats:

| Artifact | Typical Size | Description | Primary Location |
| :--- | :---: | :--- | :--- |
| **LoRA Adapter Checkpoint** | ~50 MB | Low-rank weight update matrices ($\Delta W$). | Hugging Face Adapter Repositories |
| **16-Bit Merged Model** | ~5.2 GB | Full standalone base model with LoRA baked in ($W_{\text{base}} + \Delta W$). | Hugging Face Model Hub / Local Cache |

### Remote Storage (Hugging Face Hub)
All trained adapters and the final 16-bit merged model are permanently hosted on the Hugging Face Model Hub:
* **Adapter Repository**: [orangefabercastell/gemma-2-2b-it-pi-mono-adapter-lr1e4-r16-len2k](https://huggingface.co/orangefabercastell/gemma-2-2b-it-pi-mono-adapter-lr1e4-r16-len2k)
* **Merged Model Repository**: [orangefabercastell/gemma-2-2b-it-pi-mono-sft](https://huggingface.co/orangefabercastell/gemma-2-2b-it-pi-mono-sft)

Because weights are hosted on Hugging Face, ephemeral cloud sessions (such as Kaggle or Google Colab) can be terminated immediately after upload without data loss.

### How to Download Weights Locally

Due to GitHub's 100 MB file limit, model weight binaries are hosted on Hugging Face Hub. To download the weights locally to your machine, use any of the methods below:

#### Method 1: Using the Included Python Utility Script
Run the automated download script included in this repository:

```bash
# Download the lightweight LoRA adapter (~50 MB)
python scripts/download_weights.py --repo_id orangefabercastell/gemma-2-2b-it-pi-mono-adapter-lr1e4-r16-len2k --local_dir ./checkpoints/adapter

# Download the full 16-bit merged model (~5.2 GB)
python scripts/download_weights.py --repo_id orangefabercastell/gemma-2-2b-it-pi-mono-sft --local_dir ./checkpoints/merged_model
```

#### Method 2: Using the Hugging Face CLI
```bash
# Download LoRA adapter
huggingface-cli download orangefabercastell/gemma-2-2b-it-pi-mono-adapter-lr1e4-r16-len2k --local-dir ./checkpoints/adapter

# Download full standalone merged model
huggingface-cli download orangefabercastell/gemma-2-2b-it-pi-mono-sft --local-dir ./checkpoints/merged_model
```

#### Method 3: Direct Streaming in Python (Zero Manual Download Required)
You do not need to manually download weights to run inference. The Hugging Face `transformers` library automatically downloads and caches weights upon initial load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "orangefabercastell/gemma-2-2b-it-pi-mono-sft",
    torch_dtype="auto",
    device_map="auto"
)
```

### Local Packaging from Notebook
To create local archive files directly inside a Kaggle notebook environment:

```python
import os
import tarfile

# Archive lightweight LoRA adapter (~50 MB)
with tarfile.open("gemma2_agent_adapter_best.tar.gz", "w:gz") as tar:
    tar.add(best_run["checkpoint_path"], arcname="lora_adapter")

# Archive full 16-bit merged model (~5.2 GB)
if os.path.exists("./final_merged_model"):
    with tarfile.open("gemma2_agent_merged_16bit.tar.gz", "w:gz") as tar:
        tar.add("./final_merged_model", arcname="final_merged_model")
```

---

## Future Roadmap: Production Scaling

To advance this system toward competitive software engineering agent performance, the following upgrades are planned:

1. **Scale Base Model to 7B/9B (`Qwen 2.5 Coder 7B` or `Gemma 2 9B`)**:
   * A 4-bit quantized 7B model requires approximately 5.5 GB VRAM and fits on a standard 16 GB GPU.
   * `Qwen 2.5 Coder 7B` achieves approximately 82% on HumanEval out-of-the-box, providing a substantially stronger reasoning foundation.
2. **Scale Dataset Volume (400 to 5,000+ Multi-Turn Traces)**:
   * The `pi-mono` dataset contains over 20,000 developer turns across complex multi-turn debugging sessions.
   * Training on 3,000 to 5,000 verified traces over 3 epochs will allow the model to internalize error recovery and iterative debugging loops.
3. **Preserve Chain-of-Thought Reasoning (`include_reasoning=True`)**:
   * Incorporating `<thinking>` tokens provides the model with test-time planning compute. Models trained with Chain-of-Thought reasoning typically gain +10% to +18% on code synthesis benchmarks.
4. **Hybrid Dataset Blend (70% Traces + 30% Pure Python)**:
   * Mixing agent execution traces (70%) with curated programming datasets such as `the-stack` or `python_code_instructions_18k` (30%) prevents domain overfitting and improves algorithmic problem solving.
5. **Direct Preference Optimization (DPO on Trace Outcomes)**:
   * Real developer traces include mistaken commands and syntax errors.
   * Applying DPO on paired outcomes (successful test-passing turns versus failed attempts) penalizes hallucinated tool calls and syntax errors.

---

## Quickstart and Inference

### Method A: Load Standalone Merged Model (Recommended for Production)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "orangefabercastell/gemma-2-2b-it-pi-mono-sft"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": "Find where the database port is configured in src/server.py and update it to 5432 using workspace tools."
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
print(response)
```

### Method B: Load Base Model and Attach LoRA Adapter Dynamically

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "google/gemma-2-2b-it"
ADAPTER_ID = "orangefabercastell/gemma-2-2b-it-pi-mono-adapter-lr1e4-r16-len2k"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
```

---

## Training Details and Specifications

* **Base Architecture**: Gemma 2 2B (`unsloth/gemma-2-2b-it-bnb-4bit` pre-quantized weights)
* **Fine-Tuning Method**: 4-bit QLoRA (`r=16, alpha=32, dropout=0.05`)
* **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Optimizer**: `paged_adamw_8bit` with Cosine Warmup Decay
* **Effective Batch Size**: 8 (1 per-device with 8 gradient accumulation steps)
* **Sequence Length**: 2,048 tokens with progressive context trimming
* **Tracking**: TrackIO (`pi-mono-sft-sweep`)
