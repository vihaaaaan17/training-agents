# 0002: Inline Tool Call Serialization for Gemma 2 SFT

## Context
Standard Hugging Face chat templates for `google/gemma-2-2b-it` only render `message["role"]` and `message["content"]`, ignoring `message["tool_calls"]`. When trace ingestion pipelines place parsed tool calls into `message["tool_calls"]` while omitting them from `content`, the resulting completions trained by SFT contain only conversational pleasantries, completely dropping the tool names and arguments from the training loss. Consequently, the model fails to invoke tools on standardized benchmarks like BFCL and ToolBench.

## Decision
We serialize all tool calls directly into the `content` string of assistant turns as canonical XML-delimited markup:
```xml
<tool_call>
{"name": "tool_name", "arguments": {"param1": "val1"}}
</tool_call>
```
Additionally, we prepend all training and evaluation contexts with a standardized system turn defining the available tool schemas and the `<tool_call>` protocol. To ensure out-of-domain generalization beyond the 8 workspace tools in `pi-mono`, we train on a 70/30 Hybrid Trace Blend combining `pi-mono` coding traces with multi-domain API tool-calling traces.

## Consequences
- **Positive**: Guarantees tool names and arguments are resident in the token stream and trained by completion-only loss regardless of tokenizer chat-template limitations.
- **Positive**: Enables unambiguous regex and AST parsing of tool actions during inference.
- **Trade-off**: Requires explicit system prompt injection and parsing wrappers across both training and evaluation pipelines.
