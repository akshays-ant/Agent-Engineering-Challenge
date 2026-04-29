# Helios RFP Agent — Architecture Plan (Agent SDK, Python)

**Goal:** RFP questionnaire in → structured JSON draft out in <15 min, with per-answer
sources, confidence scores, and cross-answer consistency flags.

**Stack:** `claude-agent-sdk` (Python). One orchestrator process. Custom tools served
via an in-process SDK MCP server (`create_sdk_mcp_server`). No subagents needed for
v1 — a single agent loop with the right tools and prompt is enough to hit the brief.

---

## 1. Pipeline (4 stages, 1 agent loop + 1 review pass)

```
RFP JSON ─▶ [PARSE] ─▶ [RETRIEVE+DRAFT per Q] ─▶ [REVIEW all drafts] ─▶ [EXPORT JSON]
                          (agent loop w/ tools)      (one stateless call)
```

| Stage | How | Model | Why |
|---|---|---|---|
| **Parse** | Single `query()` call, no tools, JSON-mode prompt → `[{id, text, categories[], sub_parts[]}]` | Haiku | Cheap, deterministic. Splits multi-part questions (RFP-003 Q4). |
| **Retrieve + Draft** | One `ClaudeSDKClient` session **per question**, run with `asyncio.gather`. Agent has `search_kb`, `get_document`, `list_compliance_certifications`. Loops until it emits a `DraftAnswer` JSON block. | Sonnet | Tool-use loop is the core scoring axis. Per-Q isolation = clean citations + parallelism. |
| **Review** | One `query()` call sees **all** drafts together. Checks: (a) contradictions in dates/prices/counts, (b) every claim has ≥1 source, (c) tone uniform. Emits `flags[]` + optional `rewrite`. | Sonnet | Holistic pass is what the customer pain-point demands ("answers contradict each other"). |
| **Export** | Pure Python. Merge drafts + review flags → `output.json`. | — | |

> **Key design call:** Retrieve+Draft is **fan-out per question**, not one giant context.
> Keeps each agent's KB results focused, makes citations trivially traceable, and lets
> a 50-question RFP finish in ~wall-clock-of-one-question.

---

## 2. Tool contracts (in `kb_tools.py`)

All tools are **mock**, keyword-ranked over local JSON. Swapping to real
Confluence/vector search later changes only `_build_index()` / `_score()`.

| Tool | Input | Output | Notes |
|---|---|---|---|
| `search_kb` | `{query, category, top_k}` | `[{doc_id, title, category, relevance, snippet}]` | Primary retrieval. `category` filter from Parse stage narrows the pool. |
| `get_document` | `{doc_id}` | full JSON record | When snippet is truncated and agent needs the authoritative number. |
| `list_compliance_certifications` | `{}` | full `compliance.json` | **Authoritative source.** Prompt steers agent here for cert questions so it doesn't cite stale past-RFP snippets. |

Registered via:
```python
kb_server = create_sdk_mcp_server(name="helios-kb", tools=[search_kb, get_document, list_compliance_certifications])
options = ClaudeAgentOptions(mcp_servers={"kb": kb_server}, allowed_tools=["mcp__kb__search_kb", ...])
```

---

## 3. Prompt design (the 40% lever)

**Drafter system prompt — key clauses:**
1. *"You MUST call `search_kb` at least once before answering. Never answer from prior knowledge."* → grounding.
2. *"For compliance questions, call `list_compliance_certifications` — past-RFP snippets may be stale."* → routes around the planted SOC 2 date conflict.
3. *"Cite every factual claim with a `doc_id`. If no KB result supports a claim, omit the claim."* → source attribution.
4. *"If KB coverage is partial or absent, set `confidence ≤ 0.4` and `needs_human_review: true` with a one-line reason."* → calibration (RFP-003 Q2 air-gapped, Q5 quantum).
5. *"Output ONLY a fenced ```json block matching DraftAnswer schema."* → structured output.

**Reviewer system prompt — key clauses:**
- *"Scan for contradictory facts across answers (dates, prices, counts, cert statuses). Emit a `flags` list with the conflicting question IDs and the discrepancy."*
- *"Do NOT re-retrieve. Work only from the drafts provided."* → keeps it cheap & deterministic.

---

## 4. Output schema

```json
{
  "rfp_id": "RFP-001",
  "generated_at": "...",
  "answers": [
    {
      "question_id": "Q1",
      "question": "...",
      "categories": ["technical"],
      "answer": "...",
      "sources": [{"doc_id": "doc-edr-architecture", "title": "..."}],
      "confidence": 0.92,
      "needs_human_review": false,
      "review_reason": null
    }
  ],
  "consistency_flags": [
    {"question_ids": ["Q2"], "issue": "SOC 2 date in past-RFP snippet (2023-08) conflicts with cert record (2024-01-15); used cert record."}
  ]
}
```

---

## 5. Mock data design (this is graded!)

| File | What's in it | Why it matters for evals |
|---|---|---|
| `kb/past_rfps.json` | 3 prior RFPs, 12 Q/A pairs across all 4 categories | Primary retrieval corpus. **Contains a deliberately stale SOC 2 date** (Globex 2023 vs cert record 2024) → consistency-reviewer test. |
| `kb/product_docs.json` | 5 docs: EDR arch, data sources, residency/encryption, MDR, deployment | Authoritative latency numbers (median 8s / p95 22s), agent footprint, EU region details. |
| `kb/compliance.json` | 6 certs w/ auditor, date, status | Includes `FedRAMP Moderate: in_process` → RFP-003 Q1 "FedRAMP High?" should answer **No** with nuance. |
| `kb/pricing.json` | SKUs, tier table, terms, **worked examples for 500/1k/5k** | Lets agent answer Q3 with exact numbers, not arithmetic. |
| `kb/company.json` | HQ, headcount, vertical counts, named references | Direct hit for company-info Qs. |
| `sample_rfps/rfp_001.json` | The 5 brief questions verbatim | Happy path. |
| `sample_rfps/rfp_002.json` | 6 mixed Qs | Second happy path, different facts. |
| `sample_rfps/rfp_003_edgecases.json` | FedRAMP High (no), air-gapped (no KB match), EUR pricing (partial), 3-in-1 multi-part, quantum roadmap (no KB match) | Every edge-case eval lives here. |

---

## 6. File map

```
helios-rfp/
├── README.md            ← this plan
├── agent.py             ← orchestrator (parse → fan-out draft → review → export)
├── kb_tools.py          ← @tool defs + create_sdk_mcp_server
├── prompts.py           ← system prompts as constants
├── kb/                  ← 5 JSON files (the knowledge base)
└── sample_rfps/         ← 3 input RFPs
```

---

## 7. Build order (your ~55 min)

| Min | Task | Owner |
|---|---|---|
| 0–5 | `pip install claude-agent-sdk`, drop these files in, smoke-test `kb_tools.py` standalone | Dev A |
| 5–20 | `agent.py`: Parse stage + Draft stage (single Q, then `asyncio.gather`) | Dev A+B |
| 20–30 | `prompts.py`: iterate drafter prompt until RFP-001 Q1–Q3 cite correctly | Dev B |
| 30–40 | Review stage + Export. Run RFP-001 end-to-end. | Dev A |
| 40–55 | Run RFP-002 + RFP-003. Fix the two things that break. | All |
| → evals | (separate track, not in this doc) | Dev C |

---

## 8. Demo talking points

- **Architecture slide:** the 4-stage diagram + "fan-out per question" rationale.
- **Live run:** `python agent.py sample_rfps/rfp_001.json` → cat `output.json`, highlight sources + confidence on Q3 pricing.
- **Show a catch:** run RFP-003, point at `needs_human_review: true` on air-gapped + quantum, and the FedRAMP "Moderate in-process, not High" nuance.
- **Retrospective bullets:** (a) keyword search is the weak link → vector store next, (b) reviewer should be able to *rewrite*, not just flag, (c) per-Q fan-out cost scales linearly — batch small Qs.
