# Spec: End-to-End Tool-Calling SFT and Out-of-Domain Benchmark Alignment

## Problem Statement

When evaluating the fine-tuned Gemma 2 2B agent on out-of-domain benchmarks (BFCL and ToolBench), the model performed worse than the base model (-50% delta on BFCL and 0% on ToolBench). Root-cause analysis revealed that:
1. The training ingestion pipeline stripped tool calls from the assistant turn `content` and assigned them to `message["tool_calls"]`, which Gemma 2's default chat template silently dropped. As a result, the SFT model trained only on natural language conversational filler rather than structured tool invocations.
2. The training sessions omitted canonical tool schemas from the prompt context, leaving the model without knowledge of how to map tool definitions to invocations.
3. The evaluation notebook omitted parameter schemas for ToolBench queries and used a prompt protocol mismatched with the model's training distribution.

Without resolving this serialization and schema alignment gap, fine-tuning on agent traces cannot yield functional tool-calling agents.

## Solution

A complete 2-step pipeline that:
1. **Fixes Training Data Serialization**: Re-engineers trace ingestion to explicitly inline tool calls into `message["content"]` using canonical XML markup (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`), prepends a standardized system prompt containing tool schemas, and trains on a 70/30 Hybrid Trace Blend (70% `pi-mono` coding traces + 30% multi-domain API tool-calling traces).
2. **Standardizes Benchmark Alignment**: Updates the evaluation benchmark suite with full parameter schemas for ToolBench and mirrors the identical `<tool_call>` system prompt protocol, enabling unambiguous parsing and measuring empirical deltas against the base model.

## User Stories

1. As an ML engineer, I want the training pipeline to serialize tool calls directly into the assistant's content tokens, so that completion-only loss trains the model to generate tool names and arguments rather than conversational apologies.
2. As an ML engineer, I want tool schemas to be injected into the session context during training, so that the model learns the causal relationship between tool definitions and tool invocations.
3. As an ML engineer, I want a 70/30 hybrid blend of coding agent traces and multi-domain API traces, so that the model generalizes beyond local filesystem tools to mathematical and REST API tool calls.
4. As an ML engineer, I want context trimming to preserve user anchor turns and rolling multi-turn history up to 2,048 tokens, so that the model learns iterative agent error correction.
5. As an ML engineer, I want hyperparameter sweep jobs to run natively in Kaggle free tier (Tesla T4) with completion-only loss, so that training fits within 16 GB VRAM with zero compute costs.
6. As an ML engineer, I want the winning checkpoint from the sweep to be automatically merged into 16-bit standalone weights, so that downstream inference does not require loading separate PEFT adapters.
7. As an evaluator, I want ToolBench benchmark prompts to include full parameter schemas, so that the model has the requisite schema definitions to generate valid arguments.
8. As an evaluator, I want both BFCL and ToolBench to evaluate against the standardized `<tool_call>` protocol, so that invocation and schema matching are measured accurately.
9. As an evaluator, I want a single unified evaluation loop that benchmarks Base vs. SFT models and guarantees VRAM purging, so that memory leaks do not disrupt multi-model benchmarking.
10. As an evaluator, I want a persistent scorecard saved as CSV and JSON, so that empirical deltas are tracked for repository documentation and publication.

## Implementation Decisions

1. **Canonical Tool Call Markup**: All tool invocations in assistant turns will be formatted as:
   ```xml
   <tool_call>
   {"name": "tool_name", "arguments": {"param1": "val1"}}
   </tool_call>
   ```
   This text is appended directly to `message["content"]` prior to applying Gemma 2's chat template.

2. **System Prompt Tool Schema Injection**: Every conversation begins with a developer turn containing the JSON schemas of all active tools and explicit instructions to invoke tools using `<tool_call>` markup.

3. **70/30 Hybrid Trace Blend**:
   - 70% from `badlogicgames/pi-mono` (multi-turn developer workspace operations: `bash`, `read`, `edit`, `write`, `grep`, `find`, `ls`).
   - 30% from general API tool datasets (`tuandunghcmut/toolbench-v1` or `Salesforce/xlam-function-calling-60k`) covering single-turn and multi-turn REST/function schemas.

4. **Kaggle-Native SFT Sweep Pipeline**:
   - Base model: `unsloth/gemma-2-2b-it-bnb-4bit` (4-bit QLoRA) or `google/gemma-2-2b-it`.
   - Loss: `completion_only_loss=True` via TRL `SFTTrainer`.
   - Sweep grid: `lr=2e-4, 1e-4`, LoRA `r=16, alpha=32`, max sequence length 2,048.
   - Best checkpoint selection: lowest held-out evaluation loss on unseen validation split.
   - Merging: 16-bit float merge on CPU, pushed to `orangefabercastell/gemma-2-2b-it-pi-mono-sft-v2`.

5. **Benchmark Ingestion & Evaluation Engine**:
   - BFCL (100 samples) and ToolBench (100 samples) formatted using the exact same system prompt and tool schema protocol.
   - ToolBench parquet ingestion updated to extract full parameter definitions from `benchmark/g1_tool` or inline API definitions.
   - Zero-breakout evaluation pipeline executing Base then SFT with strict `try...finally` VRAM teardown.

## Testing Decisions

1. **Seam 1 (Data Serialization & Schema Validation)**:
   - Validate that 100% of extracted training examples containing tool calls have non-empty `<tool_call>` tags in `completion`.
   - Validate that every prompt begins with `<start_of_turn>user\nYou are an AI agent with access to tools:` containing JSON schemas.
   - Validate token length distribution does not exceed 2,048 tokens.

2. **Seam 2 (SFT Model Inference Sanity Check)**:
   - Verify on held-out prompt that the fine-tuned model immediately outputs `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.

3. **Seam 3 (Out-of-Domain Empirical Evaluation)**:
   - Run 40-sample smoke test (20 BFCL + 20 ToolBench) to verify positive deltas on Tool Invocation Rate and Tool Match Accuracy before running full 200 samples.

## Out of Scope

- Hosting a live external API sandbox for dynamic shell execution (benchmarks evaluate static AST and argument schemas).
- Scaling to 7B/9B parameters during this iteration (maintains Gemma 2 2B target to ensure fast Kaggle free-tier execution).
- Multi-agent orchestrator frameworks (focuses strictly on model-level tool invocation capabilities).

## Further Notes

- Execution will occur across two dedicated notebooks:
  1. `kaggle-training-agent-trace-sft.ipynb`: Handles trace download, hybrid blending, SFT sweep, adapter selection, and 16-bit weight merging.
  2. `kaggle-eval-trace-benchmarks.ipynb`: Handles BFCL and ToolBench evaluation, AST parsing, and publication scorecard generation.
