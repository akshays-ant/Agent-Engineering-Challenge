# Helios RFP Responder

Starter repo for the **Applied AI Agent Engineering Challenge** — an agent that
takes an RFP questionnaire and produces a grounded, cited, structured draft
response in minutes instead of hours.

```
┌──────────┐   ┌──────────────┐   ┌───────────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────┐
│ RFP json │──▶│ Parse + Tag  │──▶│  Retrieve             │──▶│ Draft + Cite │──▶│ Self-Review  │──▶│  JSON  │
│ (Q list) │   │ (in-prompt)  │   │  tool: search_kb()    │   │ + confidence │   │ (2nd submit) │   │ export │
└──────────┘   └──────────────┘   └──────────┬────────────┘   └──────────────┘   └──────────────┘   └────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ kb.py — mock corpus │
                                  │ 14 docs, keyword    │
                                  │ scorer, top-3       │
                                  └─────────────────────┘
```

## Key design decisions

| Decision | Why |
|---|---|
| **Single agent, two tools** (`search_kb`, `submit_answers`) | Simpler than orchestrator+workers. One context window means the consistency review is free — the model already sees every draft. |
| **Structured output via tool call**, not "respond in JSON" | The `submit_answers` tool schema is enforced by the API. No regex parsing, no malformed-JSON retries. |
| **Review pass = re-prompt after first submit** | The brief calls out cross-answer contradictions explicitly. Feeding the model its own draft and asking it to fix contradictions is the cheapest reliable check. |
| **Confidence is categorical** (high/med/low) + `flags[]` | Easier to eval than a 0–1 float. `low` ⇒ must have a flag explaining what's missing. |
| **Mock KB is hand-seeded, keyword-scored** | Retrieval quality isn't the point. But the docs are *specific* (real numbers, dates, prices) so grounding assertions actually bite. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python run.py rfps/rfp_01.json          # full trace + JSON to stdout, also writes out/rfp_01.json
python run.py rfps/rfp_02.json --quiet  # JSON only (used by eval cache)
```

## Evals

```bash
deepeval test run test_evals.py        # runs all 3 RFPs through the agent, asserts on output
# or plain pytest:
pytest test_evals.py -v
```

DeepEval uses **Claude as the judge** (`AnthropicModel`), so the same
`ANTHROPIC_API_KEY` covers both agent and evals — no OpenAI key needed.

What's asserted (see `test_evals.py`):

- **Structure** — every answer has `sources[]`, confidence ∈ {high,med,low}, non-low ⇒ ≥1 source, all source ids exist in KB *(plain pytest, deterministic)*
- **Grounding** — `FaithfulnessMetric`: each answer's claims must be supported by the docs it cited; plus deterministic spot-checks for the latency + pricing numbers
- **Consistency** — `GEval` rubric over the full answers array: no contradictory numbers/dates across answers
- **Calibration** — RFP-02 Q1 (quantum crypto, no KB match) **must** be `low` + flagged; Q2/Q3 (full coverage) must NOT be low
- **Edge cases** — multi-part Q fully addressed (`GEval`); ambiguous Q never high-confidence

Results cache to `out/` so re-running evals doesn't re-run the agent. Set
`HELIOS_FORCE_RERUN=1` to bust the cache.

## Files

```
helios-rfp/
├── agent.py             # the agent loop — read this first
├── kb.py                # 14-doc mock KB + search_kb()
├── run.py               # CLI wrapper
├── rfps/
│   ├── rfp_01.json      # the 5 canonical Qs from the brief
│   ├── rfp_02.json      # edge cases: no-KB-match, ambiguous, 3-part compound
│   └── rfp_03.json      # short FS-flavored RFP
├── test_evals.py        # DeepEval/pytest eval suite (Claude-as-judge)
├── out/                 # run.py writes results here
└── requirements.txt
```

## 5-minute demo script

| | |
|---|---|
| **~1m Architecture** | Show the diagram above. Call out: single agent / two tools, submit_answers as the structured-output trick, review pass for consistency. |
| **~2m Live run** | `python run.py rfps/rfp_01.json` — narrate the `[tool_use] search_kb(...)` lines as they stream, then scroll the final JSON: point at `sources`, `confidence`, `flags`. |
| **~1m Evals** | `deepeval test run test_evals.py` (pre-run, terminal already open). Show the pass/fail table, then scroll to `test_calibration_no_kb_match_is_low` — "this is the calibration test: no KB match, agent correctly says low + flags it." Show one red if you have one — honest failures score points. |
| **~1m Retro** | What you'd do with another hour (see below). |

## Where to spend your team's 55 build minutes

This repo gets you a working baseline in ~5 min of setup. Spend the rest on:

1. **Richer KB + harder retrieval** — add near-duplicate / conflicting docs (e.g. an outdated pricing sheet) and see if the agent picks the right one. That's where real RFP pain lives.
2. **Consistency checker as code, not vibes** — extract every number/date from `answers[]` and assert they match across questions. Replaces the llm-rubric with something deterministic.
3. **Calibration eval** — generate 10 questions with known KB-coverage labels (5 covered, 5 not), measure precision/recall of `low` confidence. That's the "does it know what it doesn't know" metric.
4. **Parallelize search_kb calls** — right now the model calls them sequentially. Batch the parse step, fan out retrieval, then draft. Cuts a 50-Q RFP from minutes to seconds.

## Retro talking points (pre-baked honesty)

- Keyword search is the weakest link — a real Helios would need embeddings or BM25.
- Review pass adds ~1 extra model call per RFP; for 200-Q RFPs you'd want chunked review.
- `flags[]` is free-text — should probably be an enum so downstream tooling can route.
