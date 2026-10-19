"""ConfidentRetrievalStrategy: answer on a clear ranking, abstain on a close or empty one."""

from aw.routing import ConfidentRetrievalStrategy, PriorityRouter


def _search_from(table):
    return lambda q: table.get(q, [])


def test_confident_when_floor_and_gap_hold():
    s = ConfidentRetrievalStrategy(search=_search_from({"q": [("a", 0.9), ("b", 0.5)]}), floor=0.3, gap=0.2)
    assert s("q") == "a"
    assert s.explain("q") == "confident: a=0.9"


def test_abstains_below_floor_or_within_gap_or_empty():
    table = {"low": [("a", 0.2)], "close": [("a", 0.50), ("b", 0.45)], "none": []}
    s = ConfidentRetrievalStrategy(search=_search_from(table), floor=0.3, gap=0.1)
    assert s("low") is None and "below floor" in s.explain("low")
    assert s("close") is None and "ambiguous" in s.explain("close")
    assert s("none") is None and s.explain("none") == "abstained: no candidates"


def test_single_candidate_needs_no_gap():
    s = ConfidentRetrievalStrategy(search=_search_from({"q": [("only", 0.4)]}), floor=0.3, gap=0.5)
    assert s("q") == "only"


def test_falls_through_to_the_next_strategy_in_a_priority_router():
    cheap = ConfidentRetrievalStrategy(search=_search_from({"clear": [("x", 1.0)], "close": [("x", 0.5), ("y", 0.49)]}), gap=0.1)
    calls = []

    def expensive(q):
        calls.append(q)
        return "llm:" + q

    router = PriorityRouter([cheap, expensive])
    assert router("clear") == "x" and calls == []
    assert router("close") == "llm:close" and calls == ["close"]


def test_defaults_answer_any_nonempty_ranking():
    s = ConfidentRetrievalStrategy(search=_search_from({"q": [("a", 0.0), ("b", 0.0)]}))
    assert s("q") == "a"  # floor 0 and gap 0: the top candidate, whatever its score
