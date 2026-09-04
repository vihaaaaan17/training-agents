# 01: Canonical Inlined Tool-Call Trace Pipeline & 70/30 Hybrid Blend

**What to build:** An end-to-end data processing pipeline implemented inside the training notebook that parses raw agent traces from `badlogicgames/pi-mono`, explicitly serializes all tool calls into assistant turn `content` using canonical `<tool_call>{"name": ..., "arguments": ...}</tool_call>` markup, prepends a system prompt containing tool schemas, and blends 70% `pi-mono` traces with 30% API tool traces from ToolBench or XLAM.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Ingest raw JSONL traces from `badlogicgames/pi-mono` and parse tool calls directly into `message["content"]` as `<tool_call>` XML blocks.
- [x] Inject a canonical system prompt at the beginning of each multi-turn conversation containing the JSON schemas of active tools.
- [x] Ingest and format a 30% blend of general API tool-calling examples from `tuandunghcmut/toolbench-v1` or `Salesforce/xlam-function-calling-60k` with identical markup.
- [x] Implement progressive multi-turn context trimming (8, 4, 2 turns) with user anchor preservation ensuring all examples fit within 2,048 tokens.
- [x] Verify that 100% of extracted tool-calling examples retain `<tool_call>` tags in `completion` after `apply_chat_template`.
- [x] Create deterministic 90% train and 10% held-out evaluation dataset splits with a fixed random seed (`seed=42`).
