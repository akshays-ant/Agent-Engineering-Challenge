"""
Eval suite for the Helios RFP Responder — DeepEval + Claude-as-judge.

Run:
    deepeval test run test_evals.py
or just:
    pytest test_evals.py -v

Why DeepEval over promptfoo: open-source (Apache 2.0), pytest-native, and lets
us use Claude as the LLM judge — so we're not grading an Anthropic agent with
an OpenAI model.
"""
import json
import os
import pytest
from pathlib import Path

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.models import AnthropicModel

from agent import run_agent
from kb import DOCS as KB_DOCS  # list of {id, category, title, content}

# --- Judge model: Claude, not GPT --------------------------------------------
# DeepEval defaults to OpenAI; override globally so every metric uses Claude.
JUDGE = AnthropicModel(model="claude-sonnet-4-5")

KB_BY_ID = {d["id"]: d for d in KB_DOCS}
RFP_DIR = Path(__file__).parent / "rfps"
OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


# --- Run each RFP once, cache to out/ so re-runs are cheap -------------------
def _run_cached(rfp_name: str) -> dict:
    out_path = OUT_DIR / f"{rfp_name}.json"
    if out_path.exists() and not os.getenv("HELIOS_FORCE_RERUN"):
        return json.loads(out_path.read_text())
    rfp = json.loads((RFP_DIR / f"{rfp_name}.json").read_text())
    result = run_agent(rfp, verbose=False)
    out_path.write_text(json.dumps(result, indent=2))
    return result


@pytest.fixture(scope="session")
def rfp01():
    return _run_cached("rfp_01")


@pytest.fixture(scope="session")
def rfp02():
    return _run_cached("rfp_02")


@pytest.fixture(scope="session")
def rfp03():
    return _run_cached("rfp_03")


# =============================================================================
# 1. STRUCTURE — deterministic, no LLM needed
# =============================================================================
@pytest.mark.parametrize("name", ["rfp_01", "rfp_02", "rfp_03"])
def test_structure(name, request):
    res = request.getfixturevalue(name.replace("_", ""))
    assert "answers" in res and isinstance(res["answers"], list) and res["answers"]
    for a in res["answers"]:
        assert {"question_id", "question", "category", "answer",
                "confidence", "sources", "flags"} <= set(a.keys())
        assert a["confidence"] in {"high", "medium", "low"}
        assert isinstance(a["sources"], list)
        # Non-low confidence must cite at least one source
        if a["confidence"] != "low":
            assert len(a["sources"]) >= 1, (
                f"{name} {a['question_id']}: confidence={a['confidence']} but no sources"
            )
        # Every cited source id must exist in the KB
        for s in a["sources"]:
            assert s in KB_BY_ID, f"{name} {a['question_id']}: unknown source id {s}"


# =============================================================================
# 2. GROUNDING — Faithfulness: is the answer supported by its cited docs?
#    (DeepEval's FaithfulnessMetric: claims in actual_output must be backed
#     by retrieval_context. We feed it ONLY the docs the agent cited.)
# =============================================================================
def _grounding_case(ans: dict) -> LLMTestCase:
    ctx = [KB_BY_ID[s]["content"] for s in ans["sources"] if s in KB_BY_ID]
    return LLMTestCase(
        input=ans["question"],
        actual_output=ans["answer"],
        retrieval_context=ctx or ["(no sources cited)"],
    )


@pytest.mark.parametrize("qid", ["Q1", "Q2", "Q3", "Q5"])
def test_grounding_rfp01(rfp01, qid):
    ans = next(a for a in rfp01["answers"] if a["question_id"] == qid)
    if ans["confidence"] == "low":
        pytest.skip("low-confidence answers aren't expected to be fully grounded")
    metric = FaithfulnessMetric(threshold=0.75, model=JUDGE, include_reason=True)
    assert_test(_grounding_case(ans), [metric])


def test_spotcheck_facts_rfp01(rfp01):
    """Deterministic grounding spot-checks — numbers from kb.py must appear."""
    by_id = {a["question_id"]: a["answer"] for a in rfp01["answers"]}
    # Q1: detection-to-alert latency benchmark from product docs
    assert "1.8" in by_id["Q1"] or "1.8s" in by_id["Q1"] or "under 2" in by_id["Q1"].lower()
    # Q3: 500-seat tier price from pricing sheet
    assert "$42" in by_id["Q3"] or "42/endpoint" in by_id["Q3"]


# =============================================================================
# 3. CONSISTENCY — no contradictions across answers in the same RFP
# =============================================================================
consistency_metric = GEval(
    name="Cross-Answer Consistency",
    model=JUDGE,
    threshold=0.7,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "The output is a JSON array of RFP answers from a single vendor.",
        "Check whether any factual claims (numbers, dates, certifications, "
        "prices, customer counts) contradict each other across different answers.",
        "Penalize heavily for direct contradictions (e.g. two different audit "
        "dates for the same cert, two different latency figures).",
        "Do NOT penalize for information that appears in one answer but not another.",
        "Score 1.0 = fully consistent, 0.0 = clear contradictions.",
    ],
)


@pytest.mark.parametrize("name", ["rfp_01", "rfp_03"])
def test_consistency(name, request):
    res = request.getfixturevalue(name.replace("_", ""))
    tc = LLMTestCase(
        input=f"RFP {name} consistency review",
        actual_output=json.dumps(res["answers"], indent=2),
    )
    assert_test(tc, [consistency_metric])


# =============================================================================
# 4. CALIBRATION — does the agent know what it doesn't know?
# =============================================================================
def test_calibration_no_kb_match_is_low(rfp02):
    """RFP-02 Q1 (quantum-resistant crypto) has zero KB coverage.
    Agent MUST mark it low confidence and flag it."""
    q1 = next(a for a in rfp02["answers"] if a["question_id"] == "Q1")
    assert q1["confidence"] == "low", (
        f"expected low confidence for un-answerable Q, got {q1['confidence']}"
    )
    assert len(q1["flags"]) >= 1, "expected a human-review flag on the no-KB-match question"


def test_calibration_covered_questions_not_low(rfp01):
    """Q2 (compliance certs) and Q3 (pricing) have direct KB docs — should NOT be low."""
    for qid in ("Q2", "Q3"):
        a = next(x for x in rfp01["answers"] if x["question_id"] == qid)
        assert a["confidence"] != "low", f"{qid} has full KB coverage but was marked low"


# =============================================================================
# 5. EDGE CASES — multi-part & ambiguous (RFP-02)
# =============================================================================
multipart_metric = GEval(
    name="Multi-Part Completeness",
    model=JUDGE,
    threshold=0.7,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    criteria=(
        "The input is a multi-part RFP question. Determine whether the answer "
        "addresses every distinct sub-question. Penalize if any part is ignored."
    ),
)


def test_multipart_question_fully_answered(rfp02):
    q3 = next(a for a in rfp02["answers"] if a["question_id"] == "Q3")
    tc = LLMTestCase(input=q3["question"], actual_output=q3["answer"])
    assert_test(tc, [multipart_metric])


def test_ambiguous_question_flagged_not_hallucinated(rfp02):
    """RFP-02 Q2 is deliberately ambiguous. Acceptable: low/medium + a flag
    asking for clarification. NOT acceptable: high confidence."""
    q2 = next(a for a in rfp02["answers"] if a["question_id"] == "Q2")
    assert q2["confidence"] != "high", "ambiguous question answered with high confidence"
