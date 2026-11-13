from stunes.agent.graph import MAX_RELAX, AgentRecommender
from stunes.retrieval.graph import Conditions


def _agent() -> AgentRecommender:
    # only the pure helpers are exercised; no stores needed
    return AgentRecommender.__new__(AgentRecommender)


def test_relaxation_order_is_fixed_and_skips_inapplicable_steps():
    a = _agent()
    s = {
        "cond": Conditions(genres=["발라드"], tags=["감성", "새벽"], year_from=1990, year_to=1999),
        "relaxed": [],
    }
    assert a._next_relaxation(s) == "genre"
    s["relaxed"] = ["genre"]
    assert a._next_relaxation(s) == "year"
    s["relaxed"] = ["genre", "year"]
    assert a._next_relaxation(s) == "tag"
    s["relaxed"] = ["genre", "year", "tag"]
    assert a._next_relaxation(s) is None
    # nothing to relax at all: no genre, no year, a single tag
    assert a._next_relaxation({"cond": Conditions(tags=["감성"]), "relaxed": []}) is None


def test_loop_stops_at_the_cap_even_when_short():
    a = _agent()
    base = {"cond": Conditions(genres=["발라드"], year_from=1990), "limit": 10, "final": ["x"]}
    assert a._enough({**base, "attempts": 0, "relaxed": []}) == "relax"
    assert a._enough({**base, "attempts": MAX_RELAX, "relaxed": ["genre"]}) == "explain"
    assert (
        a._enough({**base, "attempts": 0, "relaxed": [], "final": [str(i) for i in range(10)]})
        == "explain"
    )


def test_llm_tags_switch_only_widens_tag_edges():
    from stunes.retrieval.bm25 import Exclusions
    from stunes.retrieval.graph import GraphRetriever

    q = "MATCH (s)-[:HAS_TAG]->(t:Tag) MATCH (s)-[:HAS_GENRE]->(g)"
    assert GraphRetriever(None)._q(q) == q
    assert GraphRetriever(None, llm_tags=True)._q(q) == (
        "MATCH (s)-[:HAS_TAG|HAS_LLM_TAG]->(t:Tag) MATCH (s)-[:HAS_GENRE]->(g)"
    )
    ex = Exclusions(tags=["새벽"])
    assert ex.clauses() == [{"terms": {"tags.kw": ["새벽"]}}]
    assert {"terms": {"llm_tags.kw": ["새벽"]}} in ex.clauses(llm_tags=True)
