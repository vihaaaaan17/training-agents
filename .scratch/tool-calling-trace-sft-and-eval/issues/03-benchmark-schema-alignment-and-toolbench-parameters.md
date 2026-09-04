# 03: Parameter-Complete Benchmark Ingestion for BFCL & ToolBench

**What to build:** An ingestion and schema alignment upgrade in `kaggle-eval-trace-benchmarks.ipynb` that provides complete parameter definitions (types, descriptions, required fields) for ToolBench queries and formats both BFCL and ToolBench evaluation prompts with the identical system prompt and `<tool_call>` protocol established during fine-tuning.

**Blocked by:** None (can start immediately or run in parallel with Ticket 01/02).

**Status:** ready-for-agent

- [ ] Update BFCL prompt construction to mirror the exact system prompt and `<tool_call>` schema protocol used during SFT.
- [ ] Upgrade ToolBench ingestion to extract complete JSON parameter schemas from `tuandunghcmut/toolbench-v1` (including argument names, parameter types, and required fields).
- [ ] Retain the configurable `SMOKE_TEST = True` toggle (20 BFCL + 20 ToolBench = 40 samples, ~3 min runtime) and full benchmark mode (100 BFCL + 100 ToolBench = 200 samples, ~12 min runtime).
- [ ] Upgrade the AST and regex parser to extract and validate `<tool_call>` XML blocks, markdown JSON, and functional signatures with exact parameter matching.
