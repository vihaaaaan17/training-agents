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
