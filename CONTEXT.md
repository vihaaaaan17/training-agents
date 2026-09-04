# Fine-Tuning Agents on Traces

Supervised fine-tuning (SFT) workflow for tuning lightweight open models on agentic code execution traces under Kaggle compute constraints.

## Language

**Trace**:
A recorded sequence of user turns, agent thinking turns, tool calls, and tool results from an autonomous coding agent session.
_Avoid_: Log, trajectory, transcript

**Assistant Example**:
A single prompt/completion pair derived from a trace conversation, where the completion is an assistant turn (text and/or tool calls).
_Avoid_: Training sample, datapoint

**Completion-Only Loss**:
Loss computed exclusively over the token IDs corresponding to the assistant's response, masking prompt and tool output tokens with -100.
_Avoid_: Target masking, response-only loss

**Sweep Job**:
A discrete training run exploring a specific combination of learning rate and LoRA rank/alpha, evaluated on held-out loss.
_Avoid_: Experiment trial, tuning run

**Held-Out Eval Loss**:
Validation cross-entropy loss evaluated on an unseen split of trace examples, used as the selection criterion for the best adapter.
_Avoid_: Validation score, test loss

**Tool Call Markup**:
The canonical text serialization format (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`) embedded directly inside an assistant turn's content to prevent chat template stripping.
_Avoid_: Function metadata, raw tool dict

**Tool Schema Injection**:
The process of prepending a multi-turn session with explicit JSON definitions of available tools and the invocation protocol.
_Avoid_: System prompt engineering, tool listing

**Hybrid Trace Blend**:
A curated training dataset comprising 70% workspace coding traces (`pi-mono`) and 30% general API tool-calling traces to ensure cross-domain generalization.
_Avoid_: Mixed dataset, multi-task bag

