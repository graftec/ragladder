"""A/B diff: rescued / lost / unchanged classification."""

from ragladder.eval.ab import ab_diff, first_relevant_rank


def test_first_relevant_rank():
    assert first_relevant_rank(["x", "a", "y"], {"a"}, k=5) == 2
    assert first_relevant_rank(["a"], {"a"}, k=5) == 1
    assert first_relevant_rank(["x", "y"], {"a"}, k=5) is None
    assert first_relevant_rank(["x", "y", "z", "a"], {"a"}, k=3) is None  # beyond k


def test_rescued_lost_unchanged():
    a = {
        "q1": ["x", "a"],   # rel a at rank 2
        "q2": ["b", "y"],   # rel b at rank 1
        "q3": ["c", "z"],   # rel c at rank 1
        "q4": ["x", "y"],   # rel d missed
    }
    b = {
        "q1": ["a", "x"],   # rank 1  -> rescued (2 -> 1)
        "q2": ["y", "b"],   # rank 2  -> lost (1 -> 2)
        "q3": ["c", "z"],   # rank 1  -> unchanged
        "q4": ["d", "x"],   # rank 1  -> rescued (miss -> 1)
    }
    qrels = {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}, "q4": {"d"}}
    res = ab_diff(a, b, qrels, k=5, a_name="A", b_name="B")

    assert {d.query_id for d in res.rescued} == {"q1", "q4"}
    assert {d.query_id for d in res.lost} == {"q2"}
    assert res.n_unchanged == 1
    # A-missed/B-hit is the biggest swing -> sorts first
    assert res.rescued[0].query_id == "q4"


def test_skips_unjudged_queries():
    res = ab_diff({"q1": ["a"]}, {"q1": ["a"]}, {"q1": set()}, k=5)
    assert res.n_unchanged == 0 and not res.rescued and not res.lost
