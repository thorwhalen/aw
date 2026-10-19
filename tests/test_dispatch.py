"""The cheap-first dispatch chain, with a fake search and a fake model (opsward ADR 0011)."""

from types import SimpleNamespace

from aw.dispatch import Adjudicator, DispatchChain, ir_search, skill_name, skills_dispatch_chain


def _fake_discover(table):
    """Mimics ir.discover's result: an object with `.results` of records having artifact_id and score."""

    def discover(corpus, query, **kw):
        hits = table.get(query, [])
        return SimpleNamespace(results=[SimpleNamespace(artifact_id=i, score=s) for i, s in hits])

    return discover


def test_ir_search_collapses_copies_and_keeps_order():
    discover = _fake_discover({"q": [("x@a", 10.0), ("x@b", 10.0), ("y", 7.0), ("y@c", 6.0)]})
    search = ir_search("skills", discover=discover)
    assert search("q") == [("x", 10.0), ("y", 7.0)]
    assert search("missing") == []
    assert skill_name("x@a") == "x"


def test_chain_answers_on_a_clear_ranking_without_calling_the_model():
    calls = []
    chain = DispatchChain(search={"q": [("a", 10.0), ("b", 5.0)]}.get, rel_gap=0.9, adjudicate=lambda r, c: calls.append(r) or "b")
    assert chain("q") == "a" and calls == []
    assert chain.trace("q")["layer"] == "retrieval"


def test_chain_escalates_on_ambiguity_with_the_raw_request_and_top_k():
    seen = {}

    def adjudicate(request, candidates):
        seen["request"], seen["candidates"] = request, candidates
        return candidates[1][0]

    chain = DispatchChain(search={"q": [("a", 10.0), ("b", 9.9), ("c", 1.0)]}.get, rel_gap=0.9, adjudicate=adjudicate, k_adjudicate=2)
    assert chain("q") == "b"
    assert seen == {"request": "q", "candidates": [("a", 10.0), ("b", 9.9)]}
    assert chain.trace("q")["layer"] == "adjudicate"


def test_chain_abstains_without_a_boundary_or_an_adjudicator():
    search = {"q": [("a", 10.0), ("b", 9.9)]}.get
    assert DispatchChain(search=search)("q") is None  # unconfigured: never routes
    t = DispatchChain(search=search, rel_gap=0.9).trace("q")
    assert t["layer"] == "none" and "no adjudicator" in t["reason"]


def test_adjudicator_prompt_shows_cards_and_parses_a_candidate_or_none():
    prompts = []

    def ask(p):
        prompts.append(p)
        return "I would pick `deslop` here."

    adj = Adjudicator(ask=ask, describe={"deslop": "Remove AI tells.", "ductus": "Gauge AI-ness."}.get, k=2)
    assert adj("make it sound human", [("deslop", 9.0), ("ductus", 8.9), ("other", 1.0)]) == "deslop"
    assert "- deslop: Remove AI tells." in prompts[0] and "- ductus: Gauge AI-ness." in prompts[0] and "other" not in prompts[0]
    assert prompts[0].rstrip().endswith("Request: make it sound human")
    assert Adjudicator(ask=lambda p: "NONE of these")("x", [("a", 1.0)]) is None
    assert Adjudicator(ask=lambda p: "")("x", [("a", 1.0)]) is None
    assert Adjudicator(ask=lambda p: "a")("x", []) is None


def test_skills_dispatch_chain_wires_ir_search_with_defaults():
    discover = _fake_discover({"land it": [("landing-a-branch@x", 40.0), ("landing-a-branch", 40.0), ("branch-hygiene", 20.0)]})
    chain = skills_dispatch_chain(discover=discover)
    assert chain.rel_gap == 0.75 and chain.floor == 0.0 and chain.adjudicate is None
    assert chain("land it") == "landing-a-branch"
    assert chain.trace("land it")["ranked"] == [("landing-a-branch", 40.0), ("branch-hygiene", 20.0)]
