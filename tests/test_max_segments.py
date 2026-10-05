"""Acceptance tests for the optional ``max_segments`` review-desk cap.

The duty roster accepts at most a fixed number of repair segments. The
independent oracle for short gaps *enumerates every legal segmentation*
(after deleting segments that intersect an A/B maintenance window) and every
A/B source assignment, then keeps precisely the plans whose segment count
respects the cap. From that finite set it derives:

* the minimum total cost and the deterministically adjudicated segments for
  both objectives (full recursive tie keys);
* the per-position ``certainty`` union over **every** cap-respecting
  minimum-cost plan (never the uncapped union);
* feasibility, distinguishing "no legal cover at all" from "covers exist but
  every one needs more segments than the cap".

Coverage:
* the cap biting exactly at the optimum (the uncapped winner uses one
  segment more than allowed -> cost and segments change, feasibility stays);
* deterministic adjudication changes among equal-cost plans;
* A/B maintenance windows combined with the cap;
* both objectives;
* omission / null / explicit defaults staying byte-for-byte identical to the
  legacy responses (with and without windows);
* the two explicit failure reasons on the API (409 detail-only, distinct
  sentinels), and strict 422 location payloads for bad ``max_segments``.
"""

import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    NoFeasibleRepair,
    SegmentLimitExceeded,
    solve,
)

client = TestClient(app)


# --------------------------------------------------------------------------
# Enumeration oracle: every legal segmentation and A/B source assignment
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
    def dfs(start, plan):
        if start == n:
            yield tuple(plan)
            return
        upper = min(n, start + max_len)
        for end in range(start + 1, upper + 1):
            for source in (0, 1):
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


def oracle_from_plans(plans, n, costs, fa, fb, windows_by_source,
                      max_segments):
    """Same adjudication as :func:`oracle` over a precomputed plan list."""
    fees = (fa, fb)
    pref = prefix_sums(n, costs)
    if not plans:
        return None
    feasible = [plan for plan in plans if len(plan) <= max_segments]
    if not feasible:
        return "LIMIT"

    optimum = min(plan_cost(plan, pref, fees) for plan in feasible)
    optimal = [
        plan for plan in feasible if plan_cost(plan, pref, fees) == optimum
    ]
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


def oracle(n, costs, fa, fb, max_len, windows_by_source, max_segments):
    """Return ``None`` (no cover at all), ``"LIMIT"`` (every cover exceeds the
    cap) or ``(optimum, default_segments, continuity_segments, certainty)``
    computed only from plans with ``len <= max_segments``.
    """
    blocked = windows_to_blocked(windows_by_source)
    plans = list(enumerate_plans(n, max_len, blocked))
    return oracle_from_plans(
        plans, n, costs, fa, fb, windows_by_source, max_segments)


def assert_solver_matches(
        n, costs, fa, fb, L, windows, max_segments, expected):
    """Compare both objectives against one oracle result."""
    for objective, index in (("default", 1), ("continuity", 2)):
        sol = solve(
            n, costs, fa, fb, L,
            objective=objective, unavailable=windows,
            max_segments=max_segments)
        assert sol.cost == expected[0]
        got = [(s.start, s.end, s.source) for s in sol.segments]
        assert got == expected[index]
        assert tuple(sol.certainty) == expected[3]
        assert len(sol.segments) <= max_segments
        # Every segment respects L and its own source's windows.
        blocked = windows_to_blocked(windows)
        for seg in sol.segments:
            src = 0 if seg.source == "A" else 1
            assert 1 <= seg.end - seg.start <= L
            assert all(k not in blocked[src]
                       for k in range(seg.start, seg.end))
        # certainty is objective-independent even under a cap.
        other = solve(
            n, costs, fa, fb, L,
            objective="continuity" if objective == "default" else "default",
            unavailable=windows, max_segments=max_segments)
        assert tuple(other.certainty) == expected[3]
        assert other.cost == expected[0]


# --------------------------------------------------------------------------
# Exhaustive short-gap sweep
# --------------------------------------------------------------------------


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


@pytest.mark.parametrize("profile_name", ["zero", "fees", "random"])
@pytest.mark.parametrize("n", range(1, 6))
def test_exhaustive_oracle_all_windows_and_caps(n, profile_name):
    # Enumerate each (n, L, window-subset pair) exactly once, then check all
    # eight caps and both objectives against that single plan list.
    rng = random.Random(90000 + n)
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
            plans = list(enumerate_plans(n, L, blocked))
            for max_segments in range(1, 9):
                expected = oracle_from_plans(
                    plans, n, costs, fa, fb, windows, max_segments)
                for objective, index in (
                        ("default", 1), ("continuity", 2)):
                    if expected is None:
                        with pytest.raises(NoFeasibleRepair):
                            solve(n, costs, fa, fb, L,
                                  objective=objective,
                                  unavailable=windows,
                                  max_segments=max_segments)
                        # No cover at all must never be reported as a cap.
                        with pytest.raises(NoFeasibleRepair):
                            solve(n, costs, fa, fb, L,
                                  objective=objective,
                                  unavailable=windows)
                        continue
                    if expected == "LIMIT":
                        with pytest.raises(SegmentLimitExceeded):
                            solve(n, costs, fa, fb, L,
                                  objective=objective,
                                  unavailable=windows,
                                  max_segments=max_segments)
                        continue
                    sol = solve(
                        n, costs, fa, fb, L,
                        objective=objective, unavailable=windows,
                        max_segments=max_segments)
                    assert sol.cost == expected[0]
                    got = [(s.start, s.end, s.source)
                           for s in sol.segments]
                    assert got == expected[index]
                    assert tuple(sol.certainty) == expected[3]
                    assert len(sol.segments) <= max_segments
                    for seg in sol.segments:
                        src = 0 if seg.source == "A" else 1
                        assert 1 <= seg.end - seg.start <= L
                        assert all(k not in blocked[src]
                                   for k in range(seg.start, seg.end))


def test_exhaustive_oracle_caps_above_n_equivalent_to_unbounded():
    # max_segments >= n can never bind (no plan needs more than n segments):
    # the capped response must equal the unbounded one on both objectives.
    rng = random.Random(424242)
    for n in range(1, 9):
        for _ in range(20):
            L = rng.randrange(1, n + 1)
            costs = [(rng.randrange(0, 5), rng.randrange(0, 5))
                     for _ in range(n)]
            fa, fb = rng.randrange(0, 5), rng.randrange(0, 5)
            for cap in (n, 8):
                if cap < n:
                    continue
                for objective in ("default", "continuity"):
                    capped = solve(n, costs, fa, fb, L,
                                   objective=objective, max_segments=cap)
                    uncapped = solve(n, costs, fa, fb, L,
                                     objective=objective)
                    assert capped.cost == uncapped.cost
                    assert [(s.start, s.end, s.source)
                            for s in capped.segments] == \
                        [(s.start, s.end, s.source)
                         for s in uncapped.segments]
                    assert tuple(capped.certainty) == \
                        tuple(uncapped.certainty)


# --------------------------------------------------------------------------
# Locked cases: cap biting exactly at the optimum
# --------------------------------------------------------------------------


def test_cap_bites_exactly_at_unbounded_optimum():
    # Make the cap bite: the cheapest ONE-segment cover differs in cost from
    # the unrestricted optimum that needs two segments. Fees are zero, so any
    # further split stays at the same position cost; the default adjudication
    # then minimizes segment count.
    n, L = 4, 4
    # A cheap at even positions, B cheap at odd. A 2-source cover achieves
    # cost 0 (A[0,2) + B[2,4), for example), while every one-segment cover
    # pays at least 20 for the positions where its source is expensive.
    costs = [(0, 10), (0, 10), (10, 0), (10, 0)]
    fa = fb = 0

    uncapped_expected = oracle(n, costs, fa, fb, L, ([], []), 8)
    assert uncapped_expected[0] == 0
    uncapped = solve(n, costs, fa, fb, L)
    assert uncapped.cost == 0
    assert [(s.start, s.end, s.source) for s in uncapped.segments] == \
        uncapped_expected[1]

    one_expected = oracle(n, costs, fa, fb, L, ([], []), 1)
    assert one_expected[0] == 20
    capped_1 = solve(n, costs, fa, fb, L, max_segments=1)
    assert len(capped_1.segments) == 1
    assert capped_1.cost == 20
    assert [(s.start, s.end, s.source) for s in capped_1.segments] == \
        one_expected[1]
    # Certainty comes from the capped optimal set only.
    assert tuple(capped_1.certainty) == one_expected[3]

    capped_2 = solve(n, costs, fa, fb, L, max_segments=2)
    assert capped_2.cost == uncapped.cost == 0
    assert [(s.start, s.end, s.source) for s in capped_2.segments] == \
        uncapped_expected[1]


def test_cap_just_below_minimum_required_segments_is_limit_not_infeasible():
    # L=1 forces exactly n singleton segments: always a legal cover, but a
    # cap below n cannot be met. The solver must distinguish that from "no
    # legal cover exists".
    n = 4
    costs = [(0, 0)] * n
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, 0, 0, 1, max_segments=n - 1)
    # At exactly n the same instance solves normally.
    sol = solve(n, costs, 0, 0, 1, max_segments=n)
    assert len(sol.segments) == n
    assert sol.cost == 0


def test_maintenance_window_forces_minimum_segment_count():
    # n=4, L=4, B never usable. A is down on a middle window [1,3), so the
    # unique A chain is [0,1) + [3,4): positions 1,2 must be served by B,
    # which is unavailable -> choose A down window such that the minimum
    # legal plan needs 2 segments. With only A available and A down on
    # [1,3), no cover exists at all; instead give B usable everywhere so the
    # cheapest plan still needs a split around the A window.
    n, L = 4, 4
    costs = [(0, 0)] * n
    # A down in the middle; B usable. A single A segment cannot span [0,4),
    # so every plan that uses the cheap A coverage must split.
    windows = ([(1, 3)], [])
    expected = oracle(n, costs, 0, 0, L, windows, max_segments=2)
    assert expected != "LIMIT" and expected is not None
    assert_solver_matches(n, costs, 0, 0, L, windows, 2, expected)

    # Cap 1 is still feasible (one B segment spans the gap) here; make B
    # expensive enough that the one-segment answer is B, while the 2-segment
    # answer is A-B-A.
    costs_b_expensive = [(0, 5)] * n
    one = solve(n, costs_b_expensive, 0, 0, L,
                unavailable=windows, max_segments=1)
    assert [(s.start, s.end, s.source)
            for s in one.segments] == [(0, 4, "B")]
    two = solve(n, costs_b_expensive, 0, 0, L,
                unavailable=windows, max_segments=2)
    assert two.cost < one.cost
    assert len(two.segments) == 2


def test_window_plus_cap_renders_no_cover_is_no_feasible():
    # Both sources down at the same position: no cover exists regardless of
    # the cap. Even a generous cap must report NO_FEASIBLE_REPAIR, not the
    # limit reason.
    n, L = 3, 2
    costs = [(0, 0)] * n
    windows = ([(1, 2)], [(1, 2)])
    for cap in (1, 2, 3, 8):
        with pytest.raises(NoFeasibleRepair):
            solve(n, costs, 0, 0, L, unavailable=windows, max_segments=cap)


# --------------------------------------------------------------------------
# Deterministic adjudication among equal-cost capped plans
# --------------------------------------------------------------------------


def test_same_cost_adjudication_changes_under_binding_cap():
    # All-zero costs and fees, n=4, L=3: one segment cannot cover all 4 so
    # every optimum uses at least 2 segments; the unbounded optimal set is
    # exactly the cap-2 one (cap 1 is infeasible). The deterministic tie
    # breakers decide which equal-cost plan is reported -- verified against
    # the enumeration oracle, not a guessed answer.
    n, L = 4, 3
    costs = [(0, 0)] * n
    fa = fb = 0
    uncapped = solve(n, costs, fa, fb, L)
    capped = solve(n, costs, fa, fb, L, max_segments=2)
    expected = oracle(n, costs, fa, fb, L, ([], []), 2)
    assert capped.cost == expected[0] == uncapped.cost == 0
    # Default: fewest segments, then smallest predecessor index, then A.
    assert [(s.start, s.end, s.source) for s in capped.segments] == \
        expected[1]
    # Continuity adjudication on the same equal-cost set.
    cont = solve(n, costs, fa, fb, L, objective="continuity",
                 max_segments=2)
    assert [(s.start, s.end, s.source) for s in cont.segments] == \
        expected[2]
    # Cap 1 admits no plan at all.
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, fa, fb, L, max_segments=1)


def test_certainty_only_uses_capped_optimal_set():
    # The cap changes which plans are optimal. Uncapped certainty is allowed
    # to mention a source that becomes impossible under the cap; the capped
    # labels must never inherit that.
    n, L = 3, 3
    # One-segment A costs 0; one-segment B costs 5. Splits can place B on a
    # zero-B-cost position, but paying one fee-free extra segment costs 0.
    costs = [(0, 5), (0, 0), (0, 5)]
    fa = fb = 0
    uncapped = solve(n, costs, fa, fb, L)
    assert uncapped.cost == 0
    capped = solve(n, costs, fa, fb, L, max_segments=1)
    # The only one-segment optimum is A; B must be impossible everywhere.
    assert tuple(capped.certainty) == ("A_ONLY", "A_ONLY", "A_ONLY")
    # Uncapped optimal plans include B at the middle position, so its label
    # is EITHER there; the cap genuinely narrows the set.
    assert uncapped.certainty[1] == "EITHER"


# --------------------------------------------------------------------------
# Random medium cross-checks against the enumeration oracle
# --------------------------------------------------------------------------


def random_windows(rng, n, p=0.3):
    result = []
    for _ in range(2):
        windows = []
        k = 0
        while k < n:
            if rng.random() < p:
                start = k
                while k < n and rng.random() < p:
                    k += 1
                if k > start:
                    windows.append((start, k))
            else:
                k += 1
        result.append(windows)
    return tuple(result)


@pytest.mark.parametrize("objective", ["default", "continuity"])
def test_random_short_instances_against_oracle(objective):
    rng = random.Random(20260923)
    for _ in range(1500):
        n = rng.randrange(1, 9)
        L = rng.randrange(1, n + 1)
        costs = [(rng.randrange(0, 7), rng.randrange(0, 7))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 7), rng.randrange(0, 7)
        windows = random_windows(rng, n)
        cap = rng.randrange(1, 9)
        expected = oracle(n, costs, fa, fb, L, windows, cap)
        if expected is None:
            with pytest.raises(NoFeasibleRepair):
                solve(n, costs, fa, fb, L, objective=objective,
                      unavailable=windows, max_segments=cap)
        elif expected == "LIMIT":
            with pytest.raises(SegmentLimitExceeded):
                solve(n, costs, fa, fb, L, objective=objective,
                      unavailable=windows, max_segments=cap)
        else:
            sol = solve(n, costs, fa, fb, L, objective=objective,
                        unavailable=windows, max_segments=cap)
            index = 1 if objective == "default" else 2
            assert sol.cost == expected[0]
            assert [(s.start, s.end, s.source) for s in sol.segments] == \
                expected[index]
            assert tuple(sol.certainty) == expected[3]


# --------------------------------------------------------------------------
# API behaviour
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


def test_api_cap_happy_path():
    # Feasible cap: n=3, L=1 forces 3 singletons; cap=3 admits the optimum.
    payload = _payload(
        n=3, L=1, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 1}, {"a": 1, "b": 0}, {"a": 0, "b": 1}],
        max_segments=3)
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cost", "segments", "certainty"}
    assert len(data["segments"]) == 3
    assert data["segments"] == [
        {"start": 0, "end": 1, "source": "A"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
    assert data["cost"] == 0


def test_api_cap_limit_exceeded_is_detail_only_409():
    # n=3, L=1 needs 3 segments; cap 2 cannot hold any cover.
    payload = _payload(
        n=3, L=1, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}] * 3,
        max_segments=2)
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "MAX_SEGMENTS_EXCEEDED"}

    # Cap=1 when at least two segments are forced.
    payload["max_segments"] = 1
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "MAX_SEGMENTS_EXCEEDED"}


def test_api_no_feasible_repair_distinct_from_limit():
    # Simultaneous outage: no legal cover at all, even with cap=8.
    payload = _payload(
        n=3, L=2,
        unavailable={"A": [{"start": 1, "end": 2}],
                     "B": [{"start": 1, "end": 2}]},
        max_segments=8)
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}


def test_api_cap_changes_cost_and_certainty():
    # The API response under a binding cap matches the capped optimal set, not
    # the uncapped solution, including certainty. A and B are each cheap on a
    # disjoint block; unrestricted cost is 0 (>= 2 segments), while every
    # one-segment plan pays 20.
    n, L = 4, 4
    costs = [{"a": 0, "b": 10}, {"a": 0, "b": 10},
             {"a": 10, "b": 0}, {"a": 10, "b": 0}]
    uncapped = client.post("/solve", json=_payload(
        n=n, L=L, fee_a=0, fee_b=0, costs=costs))
    capped = client.post("/solve", json=_payload(
        n=n, L=L, fee_a=0, fee_b=0, costs=costs, max_segments=1))
    assert uncapped.status_code == capped.status_code == 200
    assert uncapped.json()["cost"] == 0
    assert len(capped.json()["segments"]) == 1
    # Both one-segment sources cost exactly 20 (a genuine tie), so the single
    # adjudicated source is used but EITHER is possible at every position --
    # the capped certainty union, never borrowed from the uncapped optimum.
    assert capped.json()["cost"] == 20
    assert set(capped.json()["certainty"]) == {"EITHER"}


def test_api_omitted_null_and_legacy_byte_for_byte():
    # No max_segments field, explicit null, and a non-binding cap must all
    # equal the legacy response; same with maintenance windows present.
    rng = random.Random(31415)
    for _ in range(12):
        n = rng.randrange(1, 12)
        payload = _payload(
            n=n, L=rng.randrange(1, n + 1),
            fee_a=rng.randrange(0, 20), fee_b=rng.randrange(0, 20),
            costs=[{"a": rng.randrange(0, 20), "b": rng.randrange(0, 20)}
                   for _ in range(n)],
            objective=rng.choice(["default", "continuity"]),
        )
        legacy = client.post("/solve", json=payload)
        assert legacy.status_code == 200
        body = legacy.json()

        nulled = client.post(
            "/solve", json={**payload, "max_segments": None})
        assert nulled.status_code == 200
        assert nulled.json() == body

        # cap=8 never binds for n <= 8; for larger n cap cannot be exceeded
        # by any *returned* plan only when the optimum happens to use <= 8
        # segments, so only compare when n <= 8 here.
        if n <= 8:
            generous = client.post(
                "/solve", json={**payload, "max_segments": 8})
            assert generous.status_code == 200
            assert generous.json() == body

        # Same guarantees with windows present.
        with_windows = {
            **payload,
            "unavailable": {
                "A": [{"start": 0, "end": 1}] if n >= 1 else [],
                "B": []},
        }
        legacy_w = client.post("/solve", json=with_windows)
        if legacy_w.status_code == 200:
            nulled_w = client.post("/solve", json={
                **with_windows, "max_segments": None})
            assert nulled_w.json() == legacy_w.json()


def test_api_max_segments_with_windows():
    # A maintenance window changes which plans fit each cap.
    n, L = 4, 4
    costs = [{"a": 0, "b": 1_000_000}] * n
    base = _payload(n=n, L=L, fee_a=0, fee_b=1_000_000, costs=costs)

    no_window = client.post("/solve", json={**base, "max_segments": 1})
    assert no_window.status_code == 200
    assert no_window.json()["segments"] == [
        {"start": 0, "end": 4, "source": "A"}]

    # A down over [1,3): no single A segment spans the gap; B is ruinously
    # expensive but legal, so cap=1 still has a (B) cover and must not error.
    a_split = {
        **base,
        "unavailable": {"A": [{"start": 1, "end": 3}], "B": []},
        "max_segments": 1,
    }
    resp = client.post("/solve", json=a_split)
    assert resp.status_code == 200
    assert resp.json()["segments"][0]["source"] == "B"

    # Same window with cap 2: the cheap A-B-A style chain wins again.
    resp = client.post(
        "/solve", json={**a_split, "max_segments": 2})
    assert resp.status_code == 200
    assert resp.json()["cost"] < client.post(
        "/solve", json=a_split).json()["cost"]

    # A down at a single position [1,2) with B fully down: no legal cover
    # exists at all (position 1 can be served by nobody) -> this is the
    # unavailability failure, never MAX_SEGMENTS_EXCEEDED, at any cap.
    impossible = {
        **base,
        "unavailable": {"A": [{"start": 1, "end": 2}],
                        "B": [{"start": 0, "end": 4}]},
    }
    for cap in (1, 2, 8):
        resp = client.post(
            "/solve", json={**impossible, "max_segments": cap})
        assert resp.status_code == 409
        assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}

    # Two A-down stretches leave no single segment covering [0,4): the
    # blocked middle window [1,3) is shorter than the gap but, with B fully
    # down as well, the only legal A covers split around it. cap=1 is then
    # "covers exist but none fits" (MAX_SEGMENTS_EXCEEDED); however the
    # positions 1,2 also cannot be served by anyone, making it genuinely
    # infeasible. To isolate the pure segment-limit reason, keep B usable but
    # prohibitively expensive: a one-segment B cover always exists but a
    # sub-2-segment A-only cover does not, and the cap-1 optimum is B.
    limit_only = {
        **base,
        "unavailable": {"A": [{"start": 1, "end": 3}], "B": []},
    }
    resp_one = client.post(
        "/solve", json={**limit_only, "max_segments": 1})
    assert resp_one.status_code == 200
    assert resp_one.json()["segments"] == [
        {"start": 0, "end": 4, "source": "B"}]
    resp_two = client.post(
        "/solve", json={**limit_only, "max_segments": 2})
    assert resp_two.status_code == 200
    # Splitting lets cheap A segments return: strictly cheaper than one B.
    assert resp_two.json()["cost"] < resp_one.json()["cost"]
    assert len(resp_two.json()["segments"]) <= 2

    # Genuine segment-limit rejection with no unavailability: L=2 needs at
    # least ceil(4/2)=2 segments regardless of cost/windows.
    short_L = _payload(
        n=4, L=2, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}] * 4)
    resp = client.post("/solve", json={**short_L, "max_segments": 1})
    assert resp.status_code == 409
    assert resp.json() == {"detail": "MAX_SEGMENTS_EXCEEDED"}
    resp = client.post("/solve", json={**short_L, "max_segments": 2})
    assert resp.status_code == 200
    assert len(resp.json()["segments"]) == 2


@pytest.mark.parametrize("bad_value", [0, -1, 9, 100, "2", 1.5, True, None])
def test_api_invalid_max_segments_422_detail_only(bad_value):
    # None is *valid* (means "no limit"); handled separately below.
    if bad_value is None:
        return
    resp = client.post(
        "/solve", json=_payload(max_segments=bad_value))
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"detail"}
    assert "cost" not in body and "segments" not in body
    assert "certainty" not in body
    locations = {tuple(error["loc"]) for error in body["detail"]}
    assert ("body", "max_segments") in locations


def test_api_null_max_segments_is_unbounded():
    resp = client.post(
        "/solve", json=_payload(max_segments=None))
    legacy = client.post("/solve", json=_payload())
    assert resp.status_code == legacy.status_code == 200
    assert resp.json() == legacy.json()


def test_api_duplicate_max_segments_key_is_422():
    # Raw JSON: duplicate object keys remain a locatable 422.
    document = (
        '{"n": 3, "L": 1, "fee_a": 0, "fee_b": 0, '
        '"costs": [{"a": 0, "b": 0}, {"a": 0, "b": 0}, '
        '{"a": 0, "b": 0}], '
        '"max_segments": 3, "max_segments": 2}'
    )
    resp = client.post(
        "/solve", content=document,
        headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    locations = {tuple(error["loc"]) for error in resp.json()["detail"]}
    assert ("body", "max_segments") in locations


def test_api_unknown_field_alongside_cap_is_422():
    resp = client.post(
        "/solve", json={**_payload(), "max_segments": 3, "extra": 1})
    assert resp.status_code == 422
    assert ("body", "extra") in {
        tuple(error["loc"]) for error in resp.json()["detail"]}


def test_openapi_documents_max_segments():
    schema = app.openapi()
    gap = schema["components"]["schemas"]["GapRequest"]
    prop = gap["properties"]["max_segments"]
    # Optional nullable integer bounded to 1..8, documented via anyOf.
    variants = prop["anyOf"]
    integer_variant = next(
        variant for variant in variants if variant.get("type") == "integer")
    assert integer_variant["minimum"] == 1
    assert integer_variant["maximum"] == 8
    assert {"type": "null"} in variants
    assert prop.get("default") is None
    assert "max_segments" not in gap.get("required", [])


def test_replay_cost_under_cap():
    # Every returned capped plan replayed from the cost definition reproduces
    # the reported cost exactly, and segments tile [0, n).
    rng = random.Random(2718)
    n, L = 12, 4
    costs = [(rng.randrange(0, 50), rng.randrange(0, 50)) for _ in range(n)]
    fa, fb = 3, 4
    for cap in range(1, 9):
        for objective in ("default", "continuity"):
            try:
                sol = solve(n, costs, fa, fb, L,
                            objective=objective, max_segments=cap)
            except SegmentLimitExceeded:
                continue
            total = 0
            assert sol.segments[0].start == 0
            assert sol.segments[-1].end == n
            for a, b in zip(sol.segments, sol.segments[1:]):
                assert a.end == b.start
            for seg in sol.segments:
                src = 0 if seg.source == "A" else 1
                total += (fa if src == 0 else fb) + sum(
                    costs[k][src] for k in range(seg.start, seg.end))
                assert 1 <= seg.end - seg.start <= L
            assert total == sol.cost
            assert len(sol.segments) <= cap
