# W16: Agentify the assistant

The W16 work lives in the same codebase as W15 (one evolving repo):
**`../Wk15_EngineeringAISystems/ai-assistant`**

| Deliverable | Location |
|---|---|
| Agentic feature (cross-source research agent + independent verifier) | `src/assistant/agent/loop.py`, `agent/verifier.py` |
| README sections a–c + additional requirements | `README.md` §3 |
| Updated architecture diagram (agentic loop) | `docs/architecture.md` |
| Evaluation harness + results report | `eval/harness.py`, `eval/cases.yaml`, `eval/results/*.md` |
| Failure injection | `uv run python -m eval.harness --inject unavailable` (also `malformed`, `timeout`) |
