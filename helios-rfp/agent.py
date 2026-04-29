"""
agent.py — Helios RFP Agent orchestrator (Agent SDK, Python)

Usage:
    python agent.py sample_rfps/rfp_001.json

Pipeline: parse -> fan-out draft (per question, parallel) -> review -> export
"""
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from claude_agent_sdk import query, ClaudeSDKClient, ClaudeAgentOptions

from kb_tools import kb_server
from prompts import PARSER_SYSTEM, DRAFTER_SYSTEM, REVIEWER_SYSTEM

PARSE_MODEL = "claude-haiku-4-5"
DRAFT_MODEL = "claude-sonnet-4-5"
REVIEW_MODEL = "claude-sonnet-4-5"

KB_TOOLS = [
    "mcp__kb__search_kb",
    "mcp__kb__get_document",
    "mcp__kb__list_compliance_certifications",
]


def _extract_json(text: str):
    """Pull the first fenced or bare JSON object/array out of an LLM response."""
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    blob = m.group(1) if m else text
    return json.loads(blob)


async def _collect_text(aiter) -> str:
    """Concatenate all assistant text blocks from a query() / receive_response() stream."""
    out = []
    async for msg in aiter:
        content = getattr(msg, "content", None)
        if not content:
            continue
        for block in content:
            if getattr(block, "type", None) == "text" or hasattr(block, "text"):
                out.append(getattr(block, "text", ""))
    return "".join(out)


# ---------------------------------------------------------------- 1. PARSE
async def parse_rfp(rfp: dict) -> list[dict]:
    prompt = (
        "Categorize and (if needed) split these RFP questions.\n\n"
        + json.dumps(rfp["questions"], indent=2)
        + '\n\nReturn: [{"id": str, "text": str, "categories": [str], "sub_parts": [str]}]'
    )
    text = await _collect_text(
        query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=PARSE_MODEL,
                system_prompt=PARSER_SYSTEM,
                allowed_tools=[],
                max_turns=1,
            ),
        )
    )
    return _extract_json(text)


# ---------------------------------------------------------------- 2. DRAFT
async def draft_one(rfp: dict, q: dict) -> dict:
    options = ClaudeAgentOptions(
        model=DRAFT_MODEL,
        system_prompt=DRAFTER_SYSTEM,
        mcp_servers={"kb": kb_server},
        allowed_tools=KB_TOOLS,
        permission_mode="acceptAll",
        max_turns=8,
    )
    user = (
        f"Prospect: {rfp.get('prospect')} ({rfp.get('vertical')})\n"
        f"Question {q['id']} [{', '.join(q['categories'])}]:\n{q['text']}\n\n"
        "Retrieve from the KB, then draft."
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user)
        text = await _collect_text(client.receive_response())
    draft = _extract_json(text)
    draft.setdefault("question_id", q["id"])
    draft["question"] = q["text"]
    draft["categories"] = q["categories"]
    return draft


async def draft_all(rfp: dict, parsed: list[dict]) -> list[dict]:
    return await asyncio.gather(*(draft_one(rfp, q) for q in parsed))


# ---------------------------------------------------------------- 3. REVIEW
async def review(drafts: list[dict]) -> dict:
    prompt = "DRAFT ANSWERS:\n" + json.dumps(drafts, indent=2)
    text = await _collect_text(
        query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=REVIEW_MODEL,
                system_prompt=REVIEWER_SYSTEM,
                allowed_tools=[],
                max_turns=1,
            ),
        )
    )
    return _extract_json(text)


# ---------------------------------------------------------------- 4. EXPORT
def export(rfp: dict, drafts: list[dict], review_out: dict) -> dict:
    flagged_ids = {
        qid for f in review_out.get("consistency_flags", []) for qid in f["question_ids"]
    }
    for d in drafts:
        if d["question_id"] in flagged_ids and not d.get("needs_human_review"):
            d["needs_human_review"] = True
            d["review_reason"] = (d.get("review_reason") or "") + " [flagged by consistency reviewer]"
    return {
        "rfp_id": rfp["rfp_id"],
        "prospect": rfp.get("prospect"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "answers": drafts,
        "consistency_flags": review_out.get("consistency_flags", []),
        "suggested_rewrites": review_out.get("suggested_rewrites", []),
    }


# ---------------------------------------------------------------- main
async def run(rfp_path: str) -> dict:
    rfp = json.loads(Path(rfp_path).read_text())
    print(f"[parse]   {rfp['rfp_id']}: {len(rfp['questions'])} questions")
    parsed = await parse_rfp(rfp)

    print(f"[draft]   fanning out {len(parsed)} drafter agents...")
    drafts = await draft_all(rfp, parsed)

    print("[review]  checking cross-answer consistency...")
    review_out = await review(drafts)

    result = export(rfp, drafts, review_out)
    out_path = Path("output") / f"{rfp['rfp_id']}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[export]  wrote {out_path}")
    return result


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "sample_rfps/rfp_001.json"))
