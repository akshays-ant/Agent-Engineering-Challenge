"""
kb_tools.py — Mock knowledge-base tools for the Helios RFP agent.

Exposed to the Agent SDK as in-process MCP tools via @tool + create_sdk_mcp_server.
Everything is keyword-match over local JSON; swap for real Confluence/vector search later.
"""
import json
import re
from pathlib import Path
from claude_agent_sdk import tool, create_sdk_mcp_server

KB_DIR = Path(__file__).parent / "kb"


def _load(name: str):
    return json.loads((KB_DIR / f"{name}.json").read_text())


# Flatten all KB sources into a single searchable list of chunks
def _build_index():
    chunks = []
    for rfp in _load("past_rfps"):
        for qa in rfp["qa_pairs"]:
            chunks.append({
                "doc_id": rfp["doc_id"],
                "title": rfp["title"],
                "category": qa["category"],
                "text": qa["question"] + " " + qa["answer"],
                "snippet": qa["answer"],
            })
    for d in _load("product_docs"):
        chunks.append({
            "doc_id": d["doc_id"], "title": d["title"],
            "category": d["category"], "text": d["title"] + " " + d["content"],
            "snippet": d["content"],
        })
    for c in _load("compliance"):
        chunks.append({
            "doc_id": c["doc_id"], "title": c["certification"],
            "category": "compliance", "text": json.dumps(c),
            "snippet": json.dumps(c),
        })
    p = _load("pricing")
    chunks.append({
        "doc_id": p["doc_id"], "title": p["title"], "category": "pricing",
        "text": json.dumps(p), "snippet": json.dumps(p),
    })
    co = _load("company")
    chunks.append({
        "doc_id": co["doc_id"], "title": co["title"], "category": "company-info",
        "text": json.dumps(co), "snippet": json.dumps(co),
    })
    return chunks


_INDEX = _build_index()
_BY_ID = {c["doc_id"]: c for c in _INDEX}  # last write wins; fine for mock


def _score(query: str, text: str) -> float:
    terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    if not terms:
        return 0.0
    lt = text.lower()
    return sum(1 for t in terms if t in lt) / len(terms)


@tool(
    "search_kb",
    "Search the Helios knowledge base (past RFP answers, product docs, compliance "
    "records, pricing sheet, company fact sheet). Returns ranked snippets with doc_id "
    "for citation. Always call this before drafting an answer.",
    {
        "query": str,
        "category": str,  # technical | compliance | pricing | company-info | any
        "top_k": int,
    },
)
async def search_kb(args):
    query = args["query"]
    category = args.get("category", "any")
    top_k = args.get("top_k", 5)

    pool = _INDEX if category in ("any", "", None) else [
        c for c in _INDEX if c["category"] == category
    ]
    ranked = sorted(pool, key=lambda c: _score(query, c["text"]), reverse=True)
    hits = [c for c in ranked if _score(query, c["text"]) > 0][:top_k]

    results = [
        {
            "doc_id": h["doc_id"],
            "title": h["title"],
            "category": h["category"],
            "relevance": round(_score(query, h["text"]), 2),
            "snippet": h["snippet"][:600],
        }
        for h in hits
    ]
    return {"content": [{"type": "text", "text": json.dumps(results, indent=2)}]}


@tool(
    "get_document",
    "Retrieve the full content of a knowledge-base document by its doc_id "
    "(e.g. 'doc-edr-architecture', 'pricing-2024', 'cert-soc2').",
    {"doc_id": str},
)
async def get_document(args):
    doc_id = args["doc_id"]
    # Return raw JSON record so the agent sees authoritative fields
    for name in ("product_docs", "compliance", "past_rfps"):
        data = _load(name)
        for rec in data:
            if rec.get("doc_id") == doc_id:
                return {"content": [{"type": "text", "text": json.dumps(rec, indent=2)}]}
    for name in ("pricing", "company"):
        rec = _load(name)
        if rec.get("doc_id") == doc_id:
            return {"content": [{"type": "text", "text": json.dumps(rec, indent=2)}]}
    return {"content": [{"type": "text", "text": json.dumps({"error": f"doc_id '{doc_id}' not found"})}]}


@tool(
    "list_compliance_certifications",
    "Return the authoritative, current list of Helios compliance certifications with "
    "audit dates and status. Use this for any compliance question instead of relying "
    "on past-RFP snippets, which may be stale.",
    {},
)
async def list_compliance_certifications(args):
    return {"content": [{"type": "text", "text": json.dumps(_load("compliance"), indent=2)}]}


kb_server = create_sdk_mcp_server(
    name="helios-kb",
    version="1.0.0",
    tools=[search_kb, get_document, list_compliance_certifications],
)
