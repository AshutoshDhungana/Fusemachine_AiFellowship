# PaperPilot: RAG assistant → agent → MLOps (Fusemachines AI Fellowship W15–W17 Track B)

PaperPilot answers questions about the reference papers of my **Codebase Vulnerability Scanner** research
project (11 papers on ML/LLM-based vulnerability detection, plus the project's literature review).
The same codebase grows over three weeks:

| Week | What was added | Where |
|---|---|---|
| **W15 Task 1** | Gemini LLM integration, prompt engineering (temperature/top_p), JSON structured output, tool calling, RAG (ingestion → chunking → embeddings → Chroma), local vLLM model, Docker | `src/assistant/{llm,rag,assistant.py,tools.py}`, `Dockerfile` |
| **W15 Task 2** | Streamlit UI, W14 router → **ONNX + INT8**, async micro-batching, caching, retry / rate limit / fallback chain / graceful degradation, Docker Compose | `ui/`, `src/assistant/intent/`, `api.py`, `docker-compose.yml` |
| **W16** | Agentic **cross-source research agent with independent verification**, context engineering, evaluation harness, failure injection | `src/assistant/agent/`, `eval/` |
| **W17 B** | `uv`, MLflow tracking of prompt/config versions with traces, Evidently LLM-judge regression suite, Airflow DAG | `pyproject.toml`, `uv.lock`, `prompts/`, `mlops/` |

Architecture diagrams (W15 system, W16 loop, W17 MLOps loop): **[docs/architecture.md](docs/architecture.md)**.

---

## 1. Setup (one command with uv)

```bash
git clone <repo> && cd ai-assistant
uv sync --all-extras          # creates .venv exactly as pinned in uv.lock
cp .env.example .env          # put your GEMINI_API_KEY in it
```

Copy the papers into `data/papers/` (filenames are listed in `data/papers.yaml`), then:

```bash
uv run python -m assistant.rag.ingest                 # PDF → chunks → embeddings → data/chroma
# W14 model → ONNX (needs the artifacts/router_encoder folder written by the W14 notebook)
uv run python -m assistant.intent.export_onnx --src ../../Wk14_FineTuningTransformersforIntentClassification/artifacts/router_encoder
uv run uvicorn assistant.api:app --port 8080          # backend
uv run streamlit run ui/app.py                        # UI → http://localhost:8501
```

### Docker / deployment

```bash
docker compose up --build                  # api + ui   (Gemini primary + fallback)
docker compose --profile gpu up --build    # + vLLM serving Qwen2.5-1.5B-Instruct on the local GPU
```

* `vllm` uses `--gpu-memory-utilization 0.80 --max-model-len 8192` so it fits an **8 GB RTX 4070**. It needs Docker with the NVIDIA container toolkit (WSL2 on Windows). With `--enable-auto-tool-choice --tool-call-parser hermes` the local model also supports tool calling, so it is a real fallback for the agent.
* `./data` (papers and the persisted index) and `./artifacts` (the ONNX router) are mounted as volumes. The API ingests automatically on first start if the index is empty.
* Health: `GET /health`, counters: `GET /stats`, OpenAPI docs: `/docs`.
* **Cloud (bonus, not deployed):** the same compose file runs on an Azure VM (NC-series for vLLM) or on Azure Container Apps without the `gpu` profile. Push the two images to ACR, set `GEMINI_API_KEY` as a secret, and mount an Azure Files share at `/app/data`.

---

## 2. W15: design notes

**Prompt engineering.** `prompts/assistant_system.md` sets the grounding and citation rules. `TEMPERATURE=0.2` and `TOP_P=0.9` are the defaults: factual QA benefits from low entropy, and top_p < 1 cuts the long tail that produces invented numbers. Both can be changed per request from the API or the UI sliders. The agent verifier runs at temperature 0.

**Structured output.** `ResilientLLM.chat_json()` sends a JSON-schema `response_format` generated from a pydantic model (`AnswerOut`) and validates the response. If validation fails it asks the model to repair the JSON once. If a provider rejects `json_schema`, it degrades to `json_object`. Citations that point at chunks that were never retrieved are dropped.

**Tool calling.** OpenAI-style function calling (Gemini's OpenAI-compatible endpoint and vLLM both support it). The tools are `search_papers`, `get_chunk`, `list_papers` and `route_support_intent`. The last one is the W14 model, so the LLM can call a fine-tuned model as a tool. Every call goes through schema validation and a timeout, and returns an error string instead of raising.

**RAG pipeline.** `pypdf` extracts the text. Line-break hyphenation is fixed and the bibliography is cut, since reference lists poison retrieval. The text is split into 300-word windows with 60 words of overlap, *within a page*, so every chunk has an exact page citation (`paper:pN:cK`). Embeddings are `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX, CPU, no torch), stored in persistent Chroma. Retrieval is **hybrid**: dense top-N and BM25 top-N fused with reciprocal-rank fusion. This helps on this corpus because questions contain exact tokens such as model names and metric values.

**Model optimisation (Task 2).** The "trained model" is the W14 support-intent router (the best fine-tuned encoder). `export_onnx.py` exports it with dynamic batch and sequence axes, runs ORT graph optimisations (`ORT_ENABLE_ALL`: constant folding plus attention/GELU fusion) and applies **dynamic INT8 quantisation**. It writes `benchmark.json` comparing PyTorch fp32, ONNX fp32 and ONNX int8 on latency, size and label agreement. At serve time the API needs only `onnxruntime` and `tokenizers`, no torch, which keeps the image small. The LLMs themselves are not converted to ONNX: Gemini is a hosted API, and vLLM already applies PagedAttention, continuous batching and prefix caching, which suit a decoder better than an ONNX graph.

**Performance.** Everything is `async`. `/route` requests are coalesced by a micro-batcher (up to 32 items or 5 ms), so concurrent callers share one ONNX call. vLLM does continuous batching and prefix caching (the system prompt is shared). There is a TTL/LRU **response cache** keyed on messages + tools + params. `scripts_loadtest.py` measures p50/p95 and throughput and writes `docs/loadtest_*.json`.

**Reliability.**

| Mechanism | Where | Behaviour |
|---|---|---|
| Retry | `llm/client.py` | 3 attempts, exponential backoff with jitter, only on 429/5xx/timeout/connection errors |
| Rate limiting | client + API | token bucket per provider (stays under Gemini free-tier RPM); per-client inbound bucket on `/chat` and `/agent` → HTTP 429 + `Retry-After` |
| Fallback | client | `gemini-2.5-flash → gemini-2.5-flash-lite → local vLLM`; a circuit breaker skips a provider for 60 s after 3 consecutive failures |
| Degradation | `assistant.py`, `api.py` | all LLMs down → returns the top retrieved passages with `status="degraded"`; router missing → `/route` returns 503 while chat keeps working |
| Errors | `api.py` | pydantic input validation (422) and a global handler (500 JSON, no stack trace leaked) |

---

## 3. W16: Agentic feature: cross-source research with independent verification

**Why a fixed pipeline is not enough:** a comparison or "which approach…" question needs evidence from an *unknown number* of papers. Whether one more search is needed can only be decided after reading what the previous search returned, and whether the draft is actually supported can only be judged after it is written.

The agent (`src/assistant/agent/loop.py`) picks one tool per turn: `search_papers`, `get_chunk`, `list_papers`, `write_note`, `ask_user` or `finish`. Every tool call carries a `reason` argument, which is logged as the step's decision. **Stopping conditions:** `finish` accepted by the verifier, `ask_user`, `max_steps` (default 8), a token budget (60k), or all providers failing. The loop cannot run indefinitely. On `max_steps` it returns a partial answer built from NOTES, explicitly flagged as partial.

### a. Context-engineering technique

1. **Which:** *structured external notes* combined with *clearing tool results*, on top of *capped + re-ranked retrieval*.
2. **Where:** before every LLM call in the loop. The `NOTES` message (verified claims + chunk ids) is rewritten in place, and every raw `search_papers`/`get_chunk` result older than the two most recent is replaced by a one-line stub (`[cleared: 5 chunks for 'LineVD granularity' in linevd …]`).
3. **What problem it solves:** one search returns up to 8 chunks of ~900 characters, roughly 1.5–2k tokens. A comparison question takes 4–6 searches, so without clearing the transcript is re-sent in full every turn and prompt tokens grow quadratically with steps. The answer-relevant facts also get buried among stale, irrelevant chunks. Notes keep only what the model judged relevant, with provenance. `write_note` refuses chunk ids that were never retrieved, so the external memory cannot hold hallucinated evidence. The cost is that a fact the model forgot to note has to be re-read with `get_chunk`. Token effect: compare `avg_tokens` for `clear_tool_results=True/False` in `eval/results/`.

### b. Agentic pattern

**A single-agent loop plus one independent verifier sub-agent.** Research is sequential here: the next query depends on the last result, so parallel sub-agents would add coordination cost without a *sequential bottleneck* to remove. The corpus is small enough that NOTES + clearing prevents *context saturation* in one agent, and all tools belong to one skill (reading papers), so there is no *skill dilution*. The one structural failure a single loop cannot fix itself is the **self-verification paradox**: an agent re-reading its own draft in the same context is anchored by its own trajectory. The verifier therefore runs with **context isolation**: it sees only the draft and the full text of the cited chunks, never the search history. Its cost appears as `verifier_tokens`, and `--compare` runs the same queries with and without it (the single-agent baseline).

### c. Evaluation harness (`eval/harness.py`, no framework)

15 cases (`eval/cases.yaml`): 7 single-paper factual, 5 cross-paper comparisons, 2 ambiguous (should ask the user), 1 out-of-corpus (should report insufficient evidence). Per query it records:

* **Task completion:** status matches the case type, *all* expected papers are cited, and the key-fact keyword groups appear in the answer.
* **Tool-call correctness:** the tool exists, the arguments pass the schema, `paper_id` is valid and notes cite retrieved chunks. The expected tools must also be used, e.g. `ask_user` for ambiguous questions.
* **Trajectory length:** the number of steps, compared against a per-case "reasonable" bound: 4 for single-paper questions, 2N+1…8 for comparisons.
* **Tokens:** prompt, completion and verifier tokens for each query.
* **Failure log:**
  * **hard:** the run ended with no usable answer (crash, max_steps, budget, provider failure).
  * **soft:** it finished but failed a check.
  * **cascading soft:** it failed a check *and* an earlier step went wrong (a tool error, an empty search, a search restricted to the wrong paper), and the agent carried that forward instead of recovering.

Run `uv run python -m eval.harness --compare`. It writes `eval/results/<run>.md`: a summary table per mode, a per-query table with the trajectory (`search_papers → write_note → finish`), and the failure log.

**Results:** see `eval/results/` (the latest report is linked here after each run).

### Additional requirements

1. **Skill vs agent:** the verifier *could* have been a Skill, i.e. a checklist loaded into the main agent's context. I chose a sub-agent because a Skill would run inside the same context and inherit the anchoring on the agent's own reasoning, which is exactly what the verifier exists to avoid. `write_note` is a plain tool, not an agent, because it only needs deterministic validation.
2. **Token and cost accounting:** every `AgentResult` records prompt, completion and verifier tokens. The harness reports per-query and total tokens for the multi-agent mode versus the single-agent baseline, so the coordination cost of the verifier is visible.
3. **Failure injection:** `--inject unavailable|malformed|timeout` makes `search_papers` raise "connection refused", return garbled chunks without ids, or exceed the 20 s tool timeout. Garbled chunks are flagged `MALFORMED` and cannot be cited or noted, and `finish` with uncited evidence is rejected. A pass means the agent ends with `insufficient_evidence` or `error` and no citations, and says the tool failed. A confident answer under injection is logged as a *cascading soft* failure. Unit tests in `tests/test_agent_loop.py` cover all three modes offline.
4. **Tool vs agent boundary:** the vector store, and the W14 router, are the stateful multi-step services here: indexing, ANN search and BM25 fusion. I modelled them as **bounded tool calls**, with one request, one result, a timeout and schema-checked arguments, and not as agent-to-agent interactions. Their internal steps are deterministic and need no reasoning, and a tool boundary lets the harness judge each call as valid or invalid and inject faults at a single point. An agent-to-agent protocol would hide those steps and add a second place where the conversation could fail.

---

## 4. W17 Track B: MLOps for the agent

### a. Environment & reproducibility (uv)

Before the move to uv the project mixed `pip install` lines from three notebooks, which caused three concrete problems:

1. `chromadb` and `fastembed` pull incompatible `numpy`/`onnxruntime` versions unless both are resolved together.
2. `mlflow<3` and `evidently<0.7` are pinned because the legacy Evidently `TestSuite` API used here was removed in 0.7.
3. torch is only needed to *export* the ONNX model, so it is an optional extra (`--extra export`) and never enters the serving image.

`pyproject.toml` declares the ranges and `uv.lock` pins the exact resolution for every platform. `uv sync --all-extras` from a clean clone reproduces the environment, and the Dockerfile uses `uv sync --frozen`, so the image and the laptop are identical.

### b. Experiment tracking strategy (MLflow)

**Varied:** the system prompt (`prompts/agent_v1.md`, `agent_v2.md` and `agent_v3.md`, each with its SHA logged), `top_k`, `temperature` and `max_steps` (`mlops/configs/vN.yaml`).
**Measured:** the W16 harness metrics (completion overall and per case type, tool-call correctness, average steps, tokens, estimated cost, failure counts by class), robustness to the injected failure, and `pct_tests_passed` from the regression suite.
**Artifacts:** the prompt text, the config, the harness report, a per-case table and 2–3 representative **JSONL traces** per version: one clean success, one failure and one injected failure. Each step is recorded as `{step, tool, args, result, reasoning, tokens}` plus a summary record (`termination_reason`, `n_steps`, tokens).

```bash
uv run python -m mlops.run_experiment --all        # v1, v2, v3 → MLflow + mlops/results/run_comparison.md
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Trace-driven iteration** (confirm or replace each row with the trace ids from your run):

| Version | Failure seen in the previous version's traces | Change |
|---|---|---|
| v1 | (baseline, W16 prompt) | n/a |
| v2 | Comparison questions end after one search, so the second paper is uncited (soft failure on `papers_cited`). Tool errors and out-of-corpus questions are answered from background knowledge. | Per-entity targeted search with `paper_id`, an explicit sufficiency check after every search, and `insufficient_evidence` rules for tool failures |
| v3 | Ambiguous questions: the agent guesses a paper instead of calling `ask_user`. Single-paper questions take 5–7 steps. The verifier rejects paraphrased numbers. | A "Step 0" ambiguity check, a step budget per question type, verbatim numbers, `top_k` 5→4, temperature 0.2→0.1 |

**Result:** the comparison table is `mlops/results/run_comparison.md`. The winner is the version with the highest completion rate that passes the regression gate. The trade-off to report is completion versus average tokens and cost per query.

### c. Monitoring & regression strategy (Evidently)

* **Reference:** `mlops/regression/golden.jsonl` holds 10 fixed queries (8 answerable, including 3 comparisons, plus 1 out-of-corpus). Each has an *approved* answer produced by the best version and reviewed by hand (`python -m mlops.make_golden`).
* **Current:** the same queries run through the candidate prompt or config.
* **Checks:** two Evidently `LLMEval` judges with `BinaryClassificationPromptTemplate`, using Gemini as the judge through its OpenAI-compatible endpoint:
  * **Correctness:** does the answer contradict the reference?
  * **Completeness:** does it drop a key fact from the reference?
* **Tests:** the share of `incorrect` answers must be ≤ 10% and the share of `incomplete` answers ≤ 20%.
* **Outputs:** the suite writes `mlops/results/regression_vN.html`, logs `pct_tests_passed` to that version's MLflow run and **exits 1 on failure, which blocks promotion**. Per-case verdicts go to `regression_vN_per_case.csv` so the judge can be sanity-checked against a manual reading of the responses.
* **Action on a threshold breach:** do not promote. Diff the failing cases' traces against the reference traces, and either fix the prompt or, if the judge is wrong, correct the golden answer and record why.

```bash
uv run python -m mlops.make_golden --config mlops/configs/v3.yaml   # then review golden.jsonl
uv run python -m mlops.regression_suite --config mlops/configs/v3.yaml
```

### d. Orchestration (Airflow bonus)

`mlops/dags/nightly_regression_eval.py` runs at 02:00 every night:

1. It runs the harness for the production config and logs the run to MLflow.
2. It runs the Evidently regression suite.
3. `check_degradation` fails the DAG and posts to Slack (if `SLACK_WEBHOOK_URL` is set) when completion drops more than 10 points below the best historical run, when average tokens rise more than 30% above the median, or when `pct_tests_passed` < 1.

---

## 5. Tests

```bash
uv run pytest -q     # offline: scripted fake LLM + fake store (loop control, stop conditions, injection, clearing)
```

## Repository layout

```
src/assistant/  config.py · api.py · assistant.py (W15) · tools.py
                llm/client.py (retry, rate limit, fallback, cache)
                rag/{ingest,store}.py · intent/{export_onnx,onnx_router}.py
                agent/{loop,verifier}.py (W16)
prompts/        assistant_system.md · agent_v1..v3.md
eval/           cases.yaml · harness.py · results/
mlops/          run_experiment.py · make_golden.py · regression_suite.py · configs/ · regression/ · dags/ · results/
ui/app.py · Dockerfile · Dockerfile.ui · docker-compose.yml · scripts_loadtest.py · docs/architecture.md
```
