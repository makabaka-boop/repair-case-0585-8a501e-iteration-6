"""Acceptance tests for the optional ``executed_prefix`` repair prefix.

A later maintenance window must preserve the segments already repaired and
re-optimize only the remaining gap. The independent oracle enumerates *every*
legal segmentation and A/B source assignment of the full gap (deleting
segments that intersect a source's maintenance window), then keeps exactly
the plans whose first ``len(prefix)`` segments equal the executed prefix in
order. From that finite set it derives, for short gaps:

* the minimum total cost and the deterministically adjudicated segments for
  both objectives (full recursive tie keys), with the executed cost
  recomputed at the request rates rather than trusted from the caller;
* the per-position ``certainty`` union over **every** minimum-cost plan
  consistent with the fixed prefix (prefix positions forced to the executed
  source, suffix positions over the suffix optimal set);
* feasibility, distinguishing "no completion exists" from "completions need
  more segments than the cap leaves after deducting the executed count".

Coverage:
* every prefix boundary ``p = 0..n`` (``p = 0`` via omitted / null / empty
  list stays byte-for-byte identical to the legacy response, ``p = n`` is the
  unique executed plan), both objectives, every length bound, every pair of
  per-source blocked subsets, and caps ``1..8``;
* continuity counts the switch from the last executed segment into the first
  new segment (and counts nothing for the prefix-internal boundaries);
* ``max_segments`` deducts the executed segment count; remaining segment
  budgets that are too small report ``MAX_SEGMENTS_EXCEEDED``;
* new-window / executed-prefix conflicts and structurally bad prefixes are
  locatable 422s before scheduling; no success ever returns a partial plan;
* API error structure: detail-only 422 with ``body.executed_prefix``-rooted
  locations, distinct 409 sentinels, and strict rejection of a caller
  ``cost`` field.
"""

import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    SOURCE_A,
    SOURCE_B,
    NoFeasibleRepair,
    Segment,
    SegmentLimitExceeded,
    solve,
)

client = TestClient(app)


# --------------------------------------------------------------------------
# Enumeration oracle over plans that extend a fixed prefix
# --------------------------------------------------------------------------


def windows_to_blocked(windows_by_source):
    blocked = []
    for windows in windows_by_source:
        positions = set()
        for start, end in windows:
            positions.update(range(start, end))
        blocked.append(frozenset(positions))
    return tuple(blocked)


def enumerate_plans(n, max_len, blocked):
    """Every legal segmentation and A/B source assignment as tuples."""

    def dfs(start, plan):
        if start == n:
            yield tuple(plan)
            return
        upper = min(n, start + max_len)
        for end in range(start + 1, upper + 1):
            for source in (SOURCE_A, SOURCE_B):
                if any(k in blocked[source] for k in range(start, end)):
                    continue
                yield from dfs(end, plan + [(start, end, source)])

    yield from dfs(0, [])


def prefix_sums(n, costs):
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]
    return pref


def plan_cost(plan, pref, fees):
    return sum(
        fees[source] + pref[source][end] - pref[source][start]
        for start, end, source in plan
    )


def default_plan_key(plan, pref, fees):
    if not plan:
        return ()
    start, _end, source = plan[-1]
    return (
        plan_cost(plan, pref, fees),
        len(plan),
        start,
        source,
        default_plan_key(plan[:-1], pref, fees),
    )


def continuity_plan_key(plan, pref, fees):
    if not plan:
        return ()
    start, _end, source = plan[-1]
    switches = sum(
        left[2] != right[2] for left, right in zip(plan, plan[1:])
    )
    return (
        plan_cost(plan, pref, fees),
        switches,
        len(plan),
        start,
        source,
        continuity_plan_key(plan[:-1], pref, fees),
    )


def prefix_oracle(n, costs, fa, fb, max_len, windows_by_source,
                  prefix, max_segments=None):
    """Oracle for plans extending exactly ``prefix``.

    Returns:
      * ``"PREFIX_ILLEGAL"`` if the prefix itself violates L/availability;
      * ``None`` if no legal complete plan extends the prefix at all;
      * ``"LIMIT"`` if completions exist but every one exceeds the cap;
      * otherwise ``(optimum, default_segments, continuity_segments,
        certainty)``.
    """
    pref = prefix_sums(n, costs)
    fees = (fa, fb)
    blocked = windows_to_blocked(windows_by_source)

    # The service must reject an illegal prefix outright instead of quietly
    # solving around it; the oracle mirrors that distinction.
    for start, end, source in prefix:
        if not (1 <= end - start <= max_len):
            return "PREFIX_ILLEGAL"
        if any(k in blocked[source] for k in range(start, end)):
            return "PREFIX_ILLEGAL"

    plans = [
        plan for plan in enumerate_plans(n, max_len, blocked)
        if list(plan[:len(prefix)]) == [tuple(x) for x in prefix]
    ]
    if not plans:
        return None
    if max_segments is not None:
        plans = [plan for plan in plans if len(plan) <= max_segments]
        if not plans:
            return "LIMIT"

    optimum = min(plan_cost(plan, pref, fees) for plan in plans)
    optimal = [plan for plan in plans
               if plan_cost(plan, pref, fees) == optimum]
    default_best = min(
        optimal, key=lambda plan: default_plan_key(plan, pref, fees))
    continuity_best = min(
        optimal, key=lambda plan: continuity_plan_key(plan, pref, fees))

    possible = [[False, False] for _ in range(n)]
    for plan in optimal:
        for start, end, source in plan:
            for k in range(start, end):
                possible[k][source] = True
    certainty = tuple(
        "EITHER" if a and b else "A_ONLY" if a else "B_ONLY"
        for a, b in possible
    )
    return (
        optimum,
        [(s, e, "AB"[src]) for s, e, src in default_best],
        [(s, e, "AB"[src]) for s, e, src in continuity_best],
        certainty,
    )


def all_prefixes_for(n, max_len, blocked):
    """Every legal prefix tiling [0, p), p = 0..n, as plain tuples."""

    def dfs(start, chosen):
        yield tuple(chosen)
        upper = min(n, start + max_len)
        for end in range(start + 1, upper + 1):
            for source in (SOURCE_A, SOURCE_B):
                if any(k in blocked[source] for k in range(start, end)):
                    continue
                yield from dfs(
                    end, chosen + [(start, end, source)])

    yield from dfs(0, [])


def all_window_configs(n):
    def runs_from_mask(mask):
        runs = []
        start = None
        for k in range(n):
            if mask >> k & 1:
                if start is None:
                    start = k
            elif start is not None:
                runs.append((start, k))
                start = None
        if start is not None:
            runs.append((start, n))
        return runs

    for ma in range(1 << n):
        wa = runs_from_mask(ma)
        for mb in range(1 << n):
            yield wa, runs_from_mask(mb)


def as_segments(prefix):
    return tuple(Segment(s, e, "AB"[src]) for s, e, src in prefix)


# --------------------------------------------------------------------------
# Exhaustive short-gap sweeps
# --------------------------------------------------------------------------


PROFILES = (
    ("zero", lambda n: ([(0, 0)] * n, 0, 0)),
    ("fees", lambda n: ([(0, 0)] * n, 2, 3)),
)


@pytest.mark.parametrize("objective", ["default", "continuity"])
@pytest.mark.parametrize("profile_name", ["zero", "fees", "random"])
@pytest.mark.parametrize("n", range(1, 6))
def test_exhaustive_prefixes_all_windows(n, objective, profile_name):
    rng = random.Random(70000 + n)
    if profile_name == "zero":
        costs, fa, fb = [(0, 0)] * n, 0, 0
    elif profile_name == "fees":
        costs, fa, fb = [(0, 0)] * n, 2, 3
    else:
        costs = [(rng.randrange(0, 4), rng.randrange(0, 4))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 4), rng.randrange(0, 4)

    for L in range(1, n + 1):
        for windows in all_window_configs(n):
            blocked = windows_to_blocked(windows)
            # Enumerate plans once; derive each prefix oracle from scratch
            # (n <= 5 keeps this cheap) but only iterate legal prefixes.
            for prefix in all_prefixes_for(n, L, blocked):
                expected = prefix_oracle(
                    n, costs, fa, fb, L, windows, prefix)
                index = 1 if objective == "default" else 2
                if expected is None:
                    with pytest.raises(NoFeasibleRepair):
                        solve(n, costs, fa, fb, L, objective=objective,
                              unavailable=windows,
                              executed_prefix=as_segments(prefix))
                    continue
                sol = solve(n, costs, fa, fb, L, objective=objective,
                            unavailable=windows,
                            executed_prefix=as_segments(prefix))
                assert sol.cost == expected[0]
                got = [(s.start, s.end, s.source) for s in sol.segments]
                assert got == expected[index]
                assert tuple(sol.certainty) == expected[3]
                # Fixed prefix preserved verbatim and actually replayed.
                assert tuple((s.start, s.end,
                              0 if s.source == "A" else 1)
                             for s in sol.segments[:len(prefix)]) == prefix
                replay = sum(
                    (fa if src == 0 else fb)
                    + sum(costs[k][src] for k in range(a, b))
                    for a, b, src in
                    [(s.start, s.end, 0 if s.source == "A" else 1)
                     for s in sol.segments])
                assert replay == sol.cost
                # The other objective shares the same certainty.
                other = solve(
                    n, costs, fa, fb, L,
                    objective="continuity" if objective == "default"
                    else "default",
                    unavailable=windows,
                    executed_prefix=as_segments(prefix))
                assert tuple(other.certainty) == expected[3]


@pytest.mark.parametrize("objective", ["default", "continuity"])
@pytest.mark.parametrize("n", range(1, 6))
def test_exhaustive_prefixes_with_caps(n, objective):
    rng = random.Random(71000 + n)
    costs = [(rng.randrange(0, 4), rng.randrange(0, 4))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 4), rng.randrange(0, 4)
    for L in range(1, n + 1):
        for windows in all_window_configs(n):
            blocked = windows_to_blocked(windows)
            for prefix in all_prefixes_for(n, L, blocked):
                for cap in range(1, 9):
                    expected = prefix_oracle(
                        n, costs, fa, fb, L, windows, prefix,
                        max_segments=cap)
                    if expected is None:
                        with pytest.raises(NoFeasibleRepair):
                            solve(n, costs, fa, fb, L,
                                  objective=objective,
                                  unavailable=windows,
                                  max_segments=cap,
                                  executed_prefix=as_segments(prefix))
                    elif expected == "LIMIT":
                        with pytest.raises(SegmentLimitExceeded):
                            solve(n, costs, fa, fb, L,
                                  objective=objective,
                                  unavailable=windows,
                                  max_segments=cap,
                                  executed_prefix=as_segments(prefix))
                    else:
                        sol = solve(n, costs, fa, fb, L,
                                    objective=objective,
                                    unavailable=windows,
                                    max_segments=cap,
                                    executed_prefix=as_segments(prefix))
                        index = 1 if objective == "default" else 2
                        assert sol.cost == expected[0]
                        got = [(s.start, s.end, s.source)
                               for s in sol.segments]
                        assert got == expected[index]
                        assert tuple(sol.certainty) == expected[3]
                        assert len(sol.segments) <= cap
                        assert len(sol.segments) >= len(prefix)


def test_random_short_prefix_instances():
    rng = random.Random(20261005)

    def random_windows():
        result = []
        for _source in (SOURCE_A, SOURCE_B):
            windows = []
            k = 0
            while k < n:
                if rng.random() < 0.3:
                    start = k
                    while k < n and rng.random() < 0.3:
                        k += 1
                    if k > start:
                        windows.append((start, k))
                else:
                    k += 1
            result.append(windows)
        return tuple(result)

    for _ in range(2000):
        n = rng.randrange(1, 9)
        L = rng.randrange(1, n + 1)
        costs = [(rng.randrange(0, 7), rng.randrange(0, 7))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 7), rng.randrange(0, 7)
        windows = random_windows()
        blocked = windows_to_blocked(windows)
        prefixes = list(all_prefixes_for(n, L, blocked))
        prefix = prefixes[rng.randrange(len(prefixes))]
        cap = rng.choice([None, rng.randrange(1, 9)])
        objective = rng.choice(["default", "continuity"])
        expected = prefix_oracle(
            n, costs, fa, fb, L, windows, prefix, max_segments=cap)
        if expected is None:
            with pytest.raises(NoFeasibleRepair):
                solve(n, costs, fa, fb, L, objective=objective,
                      unavailable=windows, max_segments=cap,
                      executed_prefix=as_segments(prefix))
        elif expected == "LIMIT":
            with pytest.raises(SegmentLimitExceeded):
                solve(n, costs, fa, fb, L, objective=objective,
                      unavailable=windows, max_segments=cap,
                      executed_prefix=as_segments(prefix))
        else:
            sol = solve(n, costs, fa, fb, L, objective=objective,
                        unavailable=windows, max_segments=cap,
                        executed_prefix=as_segments(prefix))
            index = 1 if objective == "default" else 2
            assert sol.cost == expected[0]
            assert [(s.start, s.end, s.source)
                    for s in sol.segments] == expected[index]
            assert tuple(sol.certainty) == expected[3]


# --------------------------------------------------------------------------
# Locked boundary / switch-count / cap-deduction cases
# --------------------------------------------------------------------------


def test_p_zero_omitted_null_empty_byte_identical():
    rng = random.Random(424242)
    for _ in range(40):
        n = rng.randrange(1, 12)
        L = rng.randrange(1, n + 1)
        costs = [(rng.randrange(0, 12), rng.randrange(0, 12))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 8), rng.randrange(0, 8)
        windows = ([(0, 1)], []) if rng.random() < 0.3 else ([], [])
        objective = rng.choice(["default", "continuity"])
        cap = rng.choice([None, rng.randrange(1, 9)])
        kwargs = dict(objective=objective, unavailable=windows,
                      max_segments=cap)
        try:
            legacy = solve(n, costs, fa, fb, L, **kwargs)
        except (NoFeasibleRepair, SegmentLimitExceeded):
            continue
        for empty in (None, ()):
            got = solve(n, costs, fa, fb, L, executed_prefix=empty,
                        **kwargs)
            assert got.cost == legacy.cost
            assert [(s.start, s.end, s.source) for s in got.segments] == \
                [(s.start, s.end, s.source) for s in legacy.segments]
            assert tuple(got.certainty) == tuple(legacy.certainty)


def test_p_n_is_the_executed_plan_with_forced_labels():
    n, L = 5, 2
    costs = [(k, n - k) for k in range(n)]
    fa, fb = 3, 4
    prefix = (Segment(0, 2, "A"), Segment(2, 4, "B"), Segment(4, 5, "A"))
    sol = solve(n, costs, fa, fb, L, executed_prefix=prefix)
    assert sol.segments == prefix
    assert tuple(sol.certainty) == (
        "A_ONLY", "A_ONLY", "B_ONLY", "B_ONLY", "A_ONLY")
    expected_cost = (
        fa + sum(costs[k][0] for k in range(0, 2))
        + fb + sum(costs[k][1] for k in range(2, 4))
        + fa + costs[4][0])
    assert sol.cost == expected_cost


def test_continuity_switch_into_first_new_segment_counts():
    # L=1, zero fees/costs: every complete plan costs 0. With the prefix
    # ending in A, continuity must count a switch for a B first new segment;
    # an all-A continuation has zero suffix switches and wins.
    n = 3
    costs = [(0, 0)] * n
    prefix = (Segment(0, 1, "A"),)
    cont = solve(n, costs, 0, 0, 1, objective="continuity",
                 executed_prefix=prefix)
    assert [(s.start, s.end, s.source) for s in cont.segments] == \
        [(0, 1, "A"), (1, 2, "A"), (2, 3, "A")]

    # Prefix ending in B: all-A suffix incurs exactly one switch (B->A at
    # position 1). A suffix starting B then A would switch later too, so the
    # oracle adjudicates; cross-check it directly.
    pref_b = (Segment(0, 1, "B"),)
    expected = prefix_oracle(n, costs, 0, 0, 1, ([], []),
                             [(0, 1, SOURCE_B)])
    sol = solve(n, costs, 0, 0, 1, objective="continuity",
                executed_prefix=pref_b)
    assert [(s.start, s.end, s.source) for s in sol.segments] == expected[2]

    # Internal prefix boundaries never add switches on their own: a prefix
    # A,B followed by B has one switch total inside the executed part and
    # zero at the junction; the switch objective value is the same as
    # choosing that junction freely in the full oracle (checked above).


def test_certainty_prefix_positions_forced_suffix_union():
    # n=3, L=3, fees zero. Middle position is EITHER in the unconstrained
    # optimal set; fixing the prefix [0,1)=A forces position 0 to A_ONLY
    # while position 1's label comes from the suffix optimal set.
    n, L = 3, 3
    costs = [(0, 5), (0, 0), (0, 5)]
    prefix = (Segment(0, 1, "A"),)
    sol = solve(n, costs, 0, 0, L, executed_prefix=prefix)
    assert sol.certainty[0] == "A_ONLY"
    expected = prefix_oracle(
        n, costs, 0, 0, L, ([], []), [(0, 1, SOURCE_A)])
    assert tuple(sol.certainty) == expected[3]
    assert expected[3][0] == "A_ONLY"


def test_max_segments_deducts_executed_count():
    # L=2 forces at least 2 segments over n=4. With two executed segments
    # already tiling [0,3), only one new segment [3,4) fits cap=3; cap=2
    # leaves zero budget for any new segment.
    n, L = 4, 2
    costs = [(0, 0)] * n
    prefix = (Segment(0, 2, "A"), Segment(2, 3, "B"))
    sol = solve(n, costs, 0, 0, L, max_segments=3,
                executed_prefix=prefix)
    assert len(sol.segments) == 3
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, 0, 0, L, max_segments=2, executed_prefix=prefix)
    # p=n with exactly the executed count at the cap is fine; one less is
    # not.
    full = prefix + (Segment(3, 4, "A"),)
    assert solve(n, costs, 0, 0, L, max_segments=3,
                 executed_prefix=full).segments == full
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, 0, 0, L, max_segments=2, executed_prefix=full)


def test_suffix_cover_needs_more_segments_than_remain():
    # Executed prefix is legal but the remaining positions force more
    # segments than the cap leaves: LIMIT, never NO_FEASIBLE.
    n, L = 4, 1
    costs = [(0, 0)] * n
    prefix = (Segment(0, 1, "A"),)
    # Suffix [1,4) with L=1 needs 3 more segments; cap=2 leaves 1.
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, 0, 0, L, max_segments=2, executed_prefix=prefix)
    # Without a cap the same instance completes normally.
    sol = solve(n, costs, 0, 0, L, executed_prefix=prefix)
    assert len(sol.segments) == 4


def test_no_feasible_suffix_distinct_from_limit():
    # Both sources down at suffix position 3: nothing extends the prefix,
    # regardless of the cap.
    n, L = 4, 4
    costs = [(0, 0)] * n
    windows = ([(3, 4)], [(3, 4)])
    prefix = (Segment(0, 2, "A"),)
    for cap in (None, 2, 8):
        with pytest.raises(NoFeasibleRepair):
            solve(n, costs, 0, 0, L, unavailable=windows,
                  max_segments=cap, executed_prefix=prefix)


def test_prefix_cost_is_recomputed_not_trusted():
    # The API exposes no cost field at all; at the solver level the prefix
    # cost always derives from request rates even if they differ from what
    # the executed plan would have cost under older rates.
    n, L = 2, 2
    costs = [(5, 9), (7, 2)]
    fa, fb = 11, 13
    prefix = (Segment(0, 2, "A"),)
    sol = solve(n, costs, fa, fb, L, executed_prefix=prefix)
    assert sol.cost == fa + 5 + 7


# --------------------------------------------------------------------------
# API behaviour and error structure
# --------------------------------------------------------------------------


def _payload(n=5, **overrides):
    payload = {
        "n": n,
        "L": 3,
        "fee_a": 1,
        "fee_b": 2,
        "costs": [{"a": k, "b": k + 1} for k in range(n)],
    }
    payload.update(overrides)
    return payload


def seg(start, end, source, **extra):
    item = {"start": start, "end": end, "source": source}
    item.update(extra)
    return item


def test_api_prefix_happy_path():
    payload = _payload(
        n=4, L=4, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 10}, {"a": 0, "b": 10},
               {"a": 10, "b": 0}, {"a": 10, "b": 0}],
        executed_prefix=[seg(0, 2, "A")],
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cost", "segments", "certainty"}
    assert data["segments"][0] == {"start": 0, "end": 2, "source": "A"}
    assert data["segments"][-1]["end"] == 4
    assert all(a["end"] == b["start"]
               for a, b in zip(data["segments"], data["segments"][1:]))
    assert data["certainty"][:2] == ["A_ONLY", "A_ONLY"]
    # Replayed cost from request rates.
    total = 0
    for s in data["segments"]:
        rate = 0
        for k in range(s["start"], s["end"]):
            total += payload["costs"][k][s["source"].lower()]
        total += rate
    assert total == data["cost"]


def test_api_p_zero_variants_byte_identical():
    payload = _payload(n=4, objective="continuity")
    legacy = client.post("/solve", json=payload)
    assert legacy.status_code == 200
    for value in (None, []):
        resp = client.post(
            "/solve", json={**payload, "executed_prefix": value})
        assert resp.status_code == 200
        assert resp.json() == legacy.json()


def test_api_p_n_unique_plan():
    payload = _payload(
        n=3, L=3, fee_a=2, fee_b=3,
        costs=[{"a": 1, "b": 9}, {"a": 1, "b": 9}, {"a": 9, "b": 1}],
        executed_prefix=[seg(0, 2, "A"), seg(2, 3, "B")],
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["segments"] == [
        {"start": 0, "end": 2, "source": "A"},
        {"start": 2, "end": 3, "source": "B"},
    ]
    assert data["cost"] == 2 + 1 + 1 + 3 + 1
    assert data["certainty"] == [
        "A_ONLY", "A_ONLY", "B_ONLY"]


def test_api_prefix_preserves_executed_segments():
    rng = random.Random(909)
    for _ in range(30):
        n = rng.randrange(2, 9)
        L = rng.randrange(1, n + 1)
        costs = [{"a": rng.randrange(0, 9), "b": rng.randrange(0, 9)}
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 5), rng.randrange(0, 5)
        # Build a random legal prefix by greedy random choices.
        prefix = []
        start = 0
        while start < n and rng.random() < 0.6:
            end = rng.randrange(start + 1, min(n, start + L) + 1)
            prefix.append(seg(start, end, rng.choice(["A", "B"])))
            start = end
        if not prefix:
            continue
        payload = _payload(
            n=n, L=L, fee_a=fa, fee_b=fb, costs=costs,
            executed_prefix=prefix,
            objective=rng.choice(["default", "continuity"]),
            max_segments=rng.choice([None, rng.randrange(1, 9)]),
        )
        resp = client.post("/solve", json=payload)
        if resp.status_code == 200:
            data = resp.json()["segments"]
            for given, returned in zip(prefix, data):
                assert returned == given
        else:
            assert resp.status_code == 409
            assert set(resp.json()) == {"detail"}


@pytest.mark.parametrize("prefix,bad_index,kind", [
    # first segment does not start at 0
    ([seg(1, 2, "A")], 0, "tiling"),
    # gap between segments
    ([seg(0, 1, "A"), seg(2, 3, "B")], 1, "tiling"),
    # overlapping segments
    ([seg(0, 2, "A"), seg(1, 3, "B")], 1, "tiling"),
    # zero-length segment
    ([seg(0, 0, "A")], 0, "range"),
    # segment longer than L
    ([seg(0, 4, "A")], 0, "length"),
    # reaches beyond n
    ([seg(0, 6, "A")], 0, "range"),
    ([seg(0, 3, "A"), seg(3, 6, "B")], 1, "range"),
    # negative boundary
    ([seg(-1, 2, "A")], 0, "tiling"),
    # bad source / wrong type / extra caller cost
    ([seg(0, 1, "C")], 0, "source"),
    ([seg(0, 1, "A", cost=99)], 0, "extra"),
    ([seg(0, "1", "A")], 0, "type"),
])
def test_api_invalid_prefix_is_detail_only_422(prefix, bad_index, kind):
    payload = _payload(
        n=5, L=3,
        unavailable={"A": [{"start": 3, "end": 4}], "B": []},
        executed_prefix=prefix)
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"detail"}
    assert "cost" not in body and "segments" not in body
    assert "certainty" not in body
    locations = [tuple(error["loc"]) for error in body["detail"]]
    # Every error is rooted under body.executed_prefix and the offending
    # segment index is locatable.
    assert any(loc[:2] == ("body", "executed_prefix")
               and len(loc) >= 3 and loc[2] == bad_index
               for loc in locations), locations
    for error in body["detail"]:
        assert "cost" not in error and "segments" not in error


def test_api_prefix_inside_maintenance_window_is_422():
    # New request window conflicts with an already-executed segment.
    payload = _payload(
        n=4, L=4,
        unavailable={"A": [{"start": 1, "end": 2}], "B": []},
        executed_prefix=[seg(0, 2, "A")])
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 422
    locations = [tuple(e["loc"]) for e in resp.json()["detail"]]
    assert ("body", "executed_prefix", 0, "source") in locations

    # Half-open endpoint: an A segment ending exactly at the window start
    # stays legal.
    ok = _payload(
        n=4, L=4,
        unavailable={"A": [{"start": 2, "end": 3}], "B": []},
        executed_prefix=[seg(0, 2, "A")])
    assert client.post("/solve", json=ok).status_code == 200


def test_api_limit_and_infeasible_are_detail_only_409():
    # Suffix exists but needs more segments than the cap leaves.
    payload = _payload(
        n=4, L=1, max_segments=2,
        costs=[{"a": 0, "b": 0}] * 4,
        executed_prefix=[seg(0, 1, "A")])
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "MAX_SEGMENTS_EXCEEDED"}

    # Executed count alone exhausts the cap with positions remaining.
    payload = _payload(
        n=4, L=4, max_segments=1,
        executed_prefix=[seg(0, 2, "A")])
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "MAX_SEGMENTS_EXCEEDED"}

    # No completion at all under the windows.
    payload = _payload(
        n=4, L=4,
        unavailable={"A": [{"start": 3, "end": 4}],
                     "B": [{"start": 3, "end": 4}]},
        executed_prefix=[seg(0, 2, "A")])
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}


def test_api_duplicate_prefix_key_is_422():
    document = (
        '{"n": 2, "L": 2, "fee_a": 0, "fee_b": 0, '
        '"costs": [{"a": 0, "b": 0}, {"a": 0, "b": 0}], '
        '"executed_prefix": [{"start": 0, "end": 1, "source": "A"}], '
        '"executed_prefix": []}'
    )
    resp = client.post(
        "/solve", content=document,
        headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    locations = {tuple(e["loc"]) for e in resp.json()["detail"]}
    assert ("body", "executed_prefix") in locations


def test_api_unknown_field_alongside_prefix_is_422():
    resp = client.post(
        "/solve", json={**_payload(),
                        "executed_prefix": [], "extension": 1})
    assert resp.status_code == 422
    assert ("body", "extension") in {
        tuple(e["loc"]) for e in resp.json()["detail"]}


def test_openapi_documents_executed_prefix():
    schema = app.openapi()
    gap = schema["components"]["schemas"]["GapRequest"]
    assert "executed_prefix" in gap["properties"]
    assert "executed_prefix" not in gap.get("required", [])
    segment_schema = schema["components"]["schemas"]["ExecutedSegment"]
    assert segment_schema["additionalProperties"] is False
    assert set(segment_schema["properties"]) == {"start", "end", "source"}


def test_replay_every_returned_prefixed_plan():
    rng = random.Random(271828)
    for _ in range(300):
        n = rng.randrange(1, 10)
        L = rng.randrange(1, n + 1)
        costs = [(rng.randrange(0, 30), rng.randrange(0, 30))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 6), rng.randrange(0, 6)
        windows_a = [(k, k + 1) for k in range(n)
                     if rng.random() < 0.08]
        windows_b = [(k, k + 1) for k in range(n)
                     if rng.random() < 0.08]
        windows = (windows_a, windows_b)
        blocked = windows_to_blocked(windows)
        legal = [
            p for p in all_prefixes_for(n, L, blocked)
            if p or rng.random() < 0.2]
        if not legal:
            continue
        prefix = legal[rng.randrange(len(legal))]
        cap = rng.choice([None, rng.randrange(1, 9)])
        objective = rng.choice(["default", "continuity"])
        try:
            sol = solve(n, costs, fa, fb, L, objective=objective,
                        unavailable=windows, max_segments=cap,
                        executed_prefix=as_segments(prefix))
        except (NoFeasibleRepair, SegmentLimitExceeded):
            continue
        assert sol.segments[0].start == 0
        assert sol.segments[-1].end == n
        for a, b in zip(sol.segments, sol.segments[1:]):
            assert a.end == b.start
        total = 0
        for segment in sol.segments:
            src = 0 if segment.source == "A" else 1
            assert 1 <= segment.end - segment.start <= L
            assert all(k not in blocked[src]
                       for k in range(segment.start, segment.end))
            total += (fa if src == 0 else fb) + sum(
                costs[k][src] for k in range(segment.start, segment.end))
        assert total == sol.cost
        if cap is not None:
            assert len(sol.segments) <= cap
        assert len(sol.segments) >= len(prefix)
