"""Acceptance tests for the optional executed ``prefix`` field.

The ground station keeps the already executed repair segments and asks for a
re-optimization of the still open suffix only. The independent oracle
enumerates *every* legal segmentation and A/B source assignment (after
deleting segments that intersect a maintenance window) and then, for every
legal executed prefix and every segment cap, keeps precisely the full plans
that begin with that exact prefix. From that finite set it derives:

* the minimum total cost -- the fixed prefix repriced at the current rates,
  plus the cheapest suffix completion;
* the deterministically adjudicated segments for both objectives, including
  the source switch on the boundary between the fixed prefix and the first
  new segment (continuity);
* per-position ``certainty``: executed positions are forced to their fixed
  source; suffix positions are the source union over every minimum-cost
  completion (under the remaining cap);
* feasibility: a fixed prefix whose suffix cannot be covered is
  NO_FEASIBLE_REPAIR; completions that exist but violate the (total) segment
  cap are MAX_SEGMENTS_EXCEEDED -- including a full prefix whose own segment
  count already exceeds the cap.

Coverage also includes the boundary cases ``p == 0`` (omitted / ``null`` /
``[]`` stay byte-for-byte on the legacy response) and ``p == n``, illegal
prefix payloads (gaps, overlaps, length/window conflicts, wrong types,
duplicate JSON keys, smuggled cost fields) as detail-only 422s with locatable
``body.prefix...`` locations, the two detail-only 409 reasons, the OpenAPI
entry and the maximum-scale linear timing.
"""

import json
import random
import time

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
# Enumeration oracle
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
    """Every legal (start, end, source 0/1) plan honoring the windows."""

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


def all_legal_prefixes(plans):
    """Distinct legal prefixes: every plan cut, plus the empty prefix."""
    prefixes = {()}
    for plan in plans:
        for k in range(1, len(plan) + 1):
            prefixes.add(plan[:k])
    return sorted(prefixes, key=lambda q: (q[-1][1] if q else 0, len(q), q))


def oracle_for_prefix(plans, n, pref, fees, prefix, cap):
    """Result of fixing ``prefix`` under total-segment cap ``cap``.

    Returns ``None`` (no completion at all), ``"LIMIT"`` (completions exist
    but none respects the cap) or
    ``(optimum, default_full, continuity_full, certainty)``.
    """
    width = len(prefix)
    completions = [plan for plan in plans if plan[:width] == prefix]
    if not completions:
        return None
    feasible = [plan for plan in completions if len(plan) <= cap]
    if not feasible:
        return "LIMIT"

    optimum = min(plan_cost(plan, pref, fees) for plan in feasible)
    optimal = [plan for plan in feasible
               if plan_cost(plan, pref, fees) == optimum]
    default_best = min(optimal, key=lambda q: default_plan_key(q, pref, fees))
    continuity_best = min(
        optimal, key=lambda q: continuity_plan_key(q, pref, fees))

    possible = [[False, False] for _ in range(n)]
    # Executed positions are forced to their fixed source in every plan.
    for start, end, source in prefix:
        for k in range(start, end):
            possible[k][source] = True
    # Suffix positions: union over all minimum-cost completions.
    for plan in optimal:
        for start, end, source in plan[width:]:
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


def to_solver_prefix(prefix):
    return tuple((s, e, src) for s, e, src in prefix)


def assert_segments_respect(segments, n, L, blocked, total_cap=None):
    assert segments[0][0] == 0 and segments[-1][1] == n
    for (s, e, src), (s2, e2, _) in zip(segments, segments[1:]):
        assert e == s2
    for s, e, src in segments:
        assert 1 <= e - s <= L
        assert all(k not in blocked[src] for k in range(s, e))
    if total_cap is not None:
        assert len(segments) <= total_cap


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


# --------------------------------------------------------------------------
# Exhaustive short-gap sweep: windows x caps x prefixes x objectives
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile_name", ["zero", "fees", "random"])
@pytest.mark.parametrize("n", range(1, 5))
def test_exhaustive_oracle_all_prefixes_windows_caps(n, profile_name):
    rng = random.Random(71000 + n)
    if profile_name == "zero":
        costs, fa, fb = [(0, 0)] * n, 0, 0
    elif profile_name == "fees":
        costs, fa, fb = [(0, 0)] * n, 2, 3
    else:
        costs = [(rng.randrange(0, 4), rng.randrange(0, 4))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 4), rng.randrange(0, 4)
    fees = (fa, fb)
    pref = prefix_sums(n, costs)

    for L in range(1, n + 1):
        for windows in all_window_configs(n):
            blocked = windows_to_blocked(windows)
            plans = list(enumerate_plans(n, L, blocked))
            for prefix in all_legal_prefixes(plans):
                for cap in range(1, 9):
                    expected = oracle_for_prefix(
                        plans, n, pref, fees, prefix, cap)
                    for objective, index in (
                            ("default", 1), ("continuity", 2)):
                        if expected is None:
                            with pytest.raises(NoFeasibleRepair):
                                solve(n, costs, fa, fb, L,
                                      objective=objective,
                                      unavailable=windows,
                                      max_segments=cap,
                                      prefix=to_solver_prefix(prefix))
                            # Ignoring the cap changes nothing about the
                            # uncovered suffix.
                            with pytest.raises(NoFeasibleRepair):
                                solve(n, costs, fa, fb, L,
                                      objective=objective,
                                      unavailable=windows,
                                      prefix=to_solver_prefix(prefix))
                            continue
                        if expected == "LIMIT":
                            with pytest.raises(SegmentLimitExceeded):
                                solve(n, costs, fa, fb, L,
                                      objective=objective,
                                      unavailable=windows,
                                      max_segments=cap,
                                      prefix=to_solver_prefix(prefix))
                            continue
                        sol = solve(n, costs, fa, fb, L,
                                    objective=objective,
                                    unavailable=windows,
                                    max_segments=cap,
                                    prefix=to_solver_prefix(prefix))
                        got = [(s.start, s.end,
                                0 if s.source == "A" else 1)
                               for s in sol.segments]
                        assert sol.cost == expected[0]
                        assert [(s, e, "AB"[src])
                                for s, e, src in got] == expected[index]
                        assert tuple(sol.certainty) == expected[3]
                        assert_segments_respect(
                            got, n, L, blocked, total_cap=cap)
                        # Fixed prefix is preserved verbatim.
                        assert got[:len(prefix)] == list(prefix)
                        # Cost replays from the rates, including the fixed
                        # prefix's activation fees.
                        assert plan_cost(got, pref, fees) == sol.cost
                        # certainty is objective-independent under the fix.
                        other = solve(n, costs, fa, fb, L,
                                      objective="continuity"
                                      if objective == "default" else "default",
                                      unavailable=windows,
                                      max_segments=cap,
                                      prefix=to_solver_prefix(prefix))
                        assert tuple(other.certainty) == expected[3]
                        assert other.cost == expected[0]


# --------------------------------------------------------------------------
# Random short/medium cross-checks (n up to 8) with random legal prefixes
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
def test_random_prefix_instances_against_oracle(objective):
    rng = random.Random(20261005)
    for _ in range(1200):
        n = rng.randrange(1, 9)
        L = rng.randrange(1, n + 1)
        costs = [(rng.randrange(0, 7), rng.randrange(0, 7))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 7), rng.randrange(0, 7)
        windows = random_windows(rng, n)
        cap = rng.randrange(1, 9)
        blocked = windows_to_blocked(windows)
        plans = list(enumerate_plans(n, L, blocked))
        if not plans:
            continue
        prefix = rng.choice(all_legal_prefixes(plans))
        pref = prefix_sums(n, costs)
        expected = oracle_for_prefix(
            plans, n, pref, (fa, fb), prefix, cap)
        index = 1 if objective == "default" else 2
        if expected is None:
            with pytest.raises(NoFeasibleRepair):
                solve(n, costs, fa, fb, L, objective=objective,
                      unavailable=windows, max_segments=cap,
                      prefix=to_solver_prefix(prefix))
        elif expected == "LIMIT":
            with pytest.raises(SegmentLimitExceeded):
                solve(n, costs, fa, fb, L, objective=objective,
                      unavailable=windows, max_segments=cap,
                      prefix=to_solver_prefix(prefix))
        else:
            sol = solve(n, costs, fa, fb, L, objective=objective,
                        unavailable=windows, max_segments=cap,
                        prefix=to_solver_prefix(prefix))
            got = [(s.start, s.end, 0 if s.source == "A" else 1)
                   for s in sol.segments]
            assert sol.cost == expected[0]
            assert [(s, e, "AB"[src])
                    for s, e, src in got] == expected[index]
            assert tuple(sol.certainty) == expected[3]
            assert got[:len(prefix)] == list(prefix)
            assert plan_cost(got, pref, (fa, fb)) == sol.cost


# --------------------------------------------------------------------------
# Locked boundary/behavior cases
# --------------------------------------------------------------------------


def _lock_costs():
    # n=3, L=1, all fees zero: default ABA, continuity BBA.
    return 3, 1, [(0, 0), (1, 0), (0, 1)], 0, 0


def test_p_zero_forms_are_legacy_byte_for_byte():
    n, L, costs, fa, fb = _lock_costs()
    legacy = solve(n, costs, fa, fb, L)
    for kwargs in (dict(prefix=()),
                   dict(prefix=None)):
        sol = solve(n, costs, fa, fb, L, **kwargs)
        assert sol == legacy


def test_p_n_forces_all_labels_and_reprices():
    n, L, costs, fa, fb = _lock_costs()
    full = solve(n, costs, fa, fb, L)
    fixed = tuple(
        (s.start, s.end, 0 if s.source == "A" else 1)
        for s in full.segments)
    sol = solve(n, costs, fa, fb, L, prefix=fixed)
    assert sol.cost == full.cost
    assert tuple(sol.segments) == tuple(full.segments)
    assert tuple(sol.certainty) == tuple(
        (s.source + "_ONLY") for s in full.segments
        for _ in range(s.start, s.end))

    # Repricing: raising A's fee after the work executed must still raise the
    # reported total exactly as the rate table says -- no caller cost trusted.
    costly = solve(n, costs, 7, fb, L, prefix=fixed)
    a_segments = sum(1 for s in fixed if s[2] == 0)
    assert costly.cost == full.cost + 7 * a_segments


def test_continuity_counts_boundary_switch():
    n, L, costs, fa, fb = _lock_costs()
    # Fix A[0,1): only completion is B then A; boundary switch A->B counts,
    # total switches 2. Fixing B[0,1) yields B,B,A with one switch.
    fix_a = solve(n, costs, fa, fb, L, objective="continuity",
                  prefix=((0, 1, 0),))
    fix_b = solve(n, costs, fa, fb, L, objective="continuity",
                  prefix=((0, 1, 1),))
    assert [(s.start, s.end, s.source) for s in fix_a.segments] == [
        (0, 1, "A"), (1, 2, "B"), (2, 3, "A")]
    assert [(s.start, s.end, s.source) for s in fix_b.segments] == [
        (0, 1, "B"), (1, 2, "B"), (2, 3, "A")]
    # Both completions are optimal; labels on the suffix are identical and
    # position 0 is forced differently by the fix.
    assert fix_a.certainty[0] == "A_ONLY"
    assert fix_b.certainty[0] == "B_ONLY"
    assert tuple(fix_a.certainty[1:]) == tuple(fix_b.certainty[1:])


def test_suboptimal_fixed_prefix_changes_total_but_not_rates():
    # The fixed prefix need not be part of any unrestricted optimum: its cost
    # is recomputed and added to the best completion of the suffix.
    n, L = 4, 2
    costs = [(0, 9), (0, 9), (9, 0), (9, 0)]
    fa = fb = 0
    full = solve(n, costs, fa, fb, L)
    assert full.cost == 0
    # Force a B segment over positions 0,1 (cost 18): the suffix [2,4) is
    # then optimally B at cost 0.
    sol = solve(n, costs, fa, fb, L, prefix=((0, 2, 1),))
    assert sol.cost == 18
    assert [(s.start, s.end, s.source) for s in sol.segments] == [
        (0, 2, "B"), (2, 4, "B")]
    assert tuple(sol.certainty) == ("B_ONLY",) * n


def test_cap_subtracts_executed_segments():
    # L=1 forces singletons: n=4 needs four segments. Fixing two leaves a
    # remaining budget of two; cap 3 is exceeded, cap 4 succeeds.
    n, L = 4, 1
    costs = [(0, 0)] * n
    fixed = ((0, 1, 0), (1, 2, 1))
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, 0, 0, L, max_segments=3, prefix=fixed)
    sol = solve(n, costs, 0, 0, L, max_segments=4, prefix=fixed)
    assert len(sol.segments) == 4
    # A full prefix longer than the cap is a limit failure even at p == n.
    full_fixed = tuple((k, k + 1, k % 2) for k in range(n))
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, 0, 0, L, max_segments=3, prefix=full_fixed)
    sol2 = solve(n, costs, 0, 0, L, max_segments=4, prefix=full_fixed)
    assert sol2.cost == 0


def test_capped_certainty_suffix_only_uses_completion_optimum_set():
    # n=3, L=3: one A segment costs 0, one B segment costs 10; splitting can
    # put B on the free middle position. With one executed segment fixed at
    # [0,1) (A), cap=2 admits one more segment: both A-only and B completions
    # may tie per position -- checked against the enumeration oracle.
    n, L = 3, 3
    costs = [(0, 5), (0, 0), (0, 5)]
    fa = fb = 0
    fixed = ((0, 1, 0),)
    pref = prefix_sums(n, costs)
    plans = list(enumerate_plans(n, L, (frozenset(), frozenset())))
    expected = oracle_for_prefix(plans, n, pref, (0, 0), fixed, 2)
    sol = solve(n, costs, 0, 0, L, max_segments=2, prefix=fixed)
    assert sol.cost == expected[0]
    assert tuple(sol.certainty) == expected[3]
    assert sol.certainty[0] == "A_ONLY"  # forced executed position
    # cap=1 after one executed segment leaves no room for the suffix.
    with pytest.raises(SegmentLimitExceeded):
        solve(n, costs, 0, 0, L, max_segments=1, prefix=fixed)


def test_max_scale_prefix_timing():
    n = 200_000
    rng = random.Random(2026)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001)
    full = solve(n, costs, fa, fb, 4096)
    # Fix the entire executed portion of the unrestricted plan up to a real
    # internal boundary p (it tiles [0, p) by construction).
    idx = len(full.segments) // 2
    fixed = tuple(
        (s.start, s.end, 0 if s.source == "A" else 1)
        for s in full.segments[:idx])
    assert fixed and fixed[-1][1] < n
    for objective in ("default", "continuity"):
        t0 = time.perf_counter()
        sol = solve(n, costs, fa, fb, 4096, objective=objective,
                    prefix=fixed)
        assert time.perf_counter() - t0 < 10.0
        assert len(sol.certainty) == n
        assert sol.segments[0].start == 0
        assert sol.segments[-1].end == n

    # Capped sweep at scale: L large enough that <= 8 segments suffice.
    wide = solve(n, costs, fa, fb, 30_000, max_segments=8)
    wide_idx = len(wide.segments) // 2
    wide_fixed = tuple(
        (s.start, s.end, 0 if s.source == "A" else 1)
        for s in wide.segments[:wide_idx])
    for objective in ("default", "continuity"):
        t0 = time.perf_counter()
        sol = solve(n, costs, fa, fb, 30_000, objective=objective,
                    max_segments=8, prefix=wide_fixed)
        assert time.perf_counter() - t0 < 10.0
        assert len(sol.segments) <= 8
        assert sol.segments[0].start == 0


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


def test_api_prefix_happy_path():
    resp = client.post("/solve", json=_payload(
        prefix=[{"start": 0, "end": 2, "source": "A"}]))
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cost", "segments", "certainty"}
    assert data["segments"][0] == {"start": 0, "end": 2, "source": "A"}
    # Back-to-back tiling of the full gap.
    assert data["segments"][-1]["end"] == 5
    for a, b in zip(data["segments"], data["segments"][1:]):
        assert a["end"] == b["start"]


def test_api_omitted_null_empty_prefix_identical():
    legacy = client.post("/solve", json=_payload())
    assert legacy.status_code == 200
    for body in (_payload(prefix=None), _payload(prefix=[])):
        resp = client.post("/solve", json=body)
        assert resp.status_code == 200
        assert resp.json() == legacy.json()


def test_api_prefix_p_n_is_detail_complete():
    base = client.post("/solve", json=_payload()).json()
    resp = client.post("/solve", json=_payload(prefix=base["segments"]))
    assert resp.status_code == 200
    data = resp.json()
    assert data["cost"] == base["cost"]
    assert data["segments"] == base["segments"]
    # Every executed position is forced to its executed source.
    for seg in base["segments"]:
        for k in range(seg["start"], seg["end"]):
            assert data["certainty"][k] == seg["source"] + "_ONLY"


def test_api_suffix_no_feasible_is_detail_only_409():
    payload = {
        "n": 3, "L": 2, "fee_a": 0, "fee_b": 0,
        "costs": [{"a": 0, "b": 0}] * 3,
        "unavailable": {"A": [{"start": 1, "end": 2}],
                        "B": [{"start": 1, "end": 2}]},
        "prefix": [{"start": 0, "end": 1, "source": "A"}],
    }
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}


def test_api_remaining_budget_exceeded_is_409():
    payload = {
        "n": 4, "L": 1, "fee_a": 0, "fee_b": 0,
        "costs": [{"a": 0, "b": 0}] * 4,
        "max_segments": 3,
        "prefix": [
            {"start": 0, "end": 1, "source": "A"},
            {"start": 1, "end": 2, "source": "B"},
        ],
    }
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "MAX_SEGMENTS_EXCEEDED"}
    # Full prefix alone exceeding the cap is the same reason.
    payload["max_segments"] = 2
    payload["prefix"].append({"start": 2, "end": 3, "source": "A"})
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "MAX_SEGMENTS_EXCEEDED"}


def test_api_prefix_window_conflict_is_422_not_409():
    # A legal-looking prefix that intersects a *new* maintenance window is an
    # invalid request (422), never an infeasible instance.
    payload = _payload(unavailable={
        "A": [{"start": 1, "end": 3}], "B": []},
        prefix=[{"start": 0, "end": 3, "source": "A"}])
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"detail"}
    assert tuple(body["detail"][0]["loc"]) == ("body", "prefix", 0)
    for error in body["detail"]:
        assert "cost" not in error and "segments" not in error


def test_api_prefix_cost_cannot_be_supplied():
    # Smuggled fee/cost fields on an executed segment are rejected outright.
    for smuggled in ("cost", "fee", "price", "total"):
        resp = client.post("/solve", json=_payload(prefix=[
            {"start": 0, "end": 1, "source": "A", smuggled: 0}]))
        assert resp.status_code == 422
        locations = {tuple(e["loc"]) for e in resp.json()["detail"]}
        assert ("body", "prefix", 0, smuggled) in locations


@pytest.mark.parametrize("prefix,expected_loc", [
    # does not start at 0
    ([{"start": 1, "end": 2, "source": "A"}],
     ("body", "prefix", 0, "start")),
    # gap between segments
    ([{"start": 0, "end": 1, "source": "A"},
      {"start": 2, "end": 3, "source": "A"}],
     ("body", "prefix", 1, "start")),
    # overlap between segments
    ([{"start": 0, "end": 3, "source": "A"},
      {"start": 2, "end": 4, "source": "A"}],
     ("body", "prefix", 1, "start")),
    # zero-length segment
    ([{"start": 1, "end": 1, "source": "A"}],
     ("body", "prefix", 0, "end")),
    # end beyond n
    ([{"start": 0, "end": 6, "source": "A"}],
     ("body", "prefix", 0, "end")),
    # segment longer than L (L=3)
    ([{"start": 0, "end": 4, "source": "A"}],
     ("body", "prefix", 0, "end")),
    # unknown source
    ([{"start": 0, "end": 1, "source": "C"}],
     ("body", "prefix", 0, "source")),
    # missing source
    ([{"start": 0, "end": 1}],
     ("body", "prefix", 0, "source")),
    # wrong type inside segment
    ([{"start": "0", "end": 1, "source": "A"}],
     ("body", "prefix", 0, "start")),
    # null element
    ([None], ("body", "prefix", 0)),
    # segment not an object
    ([1], ("body", "prefix", 0)),
    # prefix not a list
    ({}, ("body", "prefix",)),
])
def test_api_illegal_prefix_is_detail_only_422(prefix, expected_loc):
    resp = client.post("/solve", json=_payload(prefix=prefix))
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"detail"}
    assert isinstance(body["detail"], list) and body["detail"]
    locations = {tuple(error["loc"]) for error in body["detail"]}
    assert expected_loc in locations
    for error in body["detail"]:
        assert "cost" not in error and "segments" not in error
        assert "certainty" not in error


def test_api_duplicate_prefix_keys_are_422():
    document = json.dumps({
        "n": 3, "L": 3, "fee_a": 0, "fee_b": 0,
        "costs": [{"a": 0, "b": 0}] * 3,
        # repeated "prefix" (identical here) must still be rejected
        "prefix": [{"start": 0, "end": 1, "source": "A"}],
    })
    document = document[:-1] + (
        ', "prefix": [{"start": 1, "end": 2, "source": "B"}]}')
    resp = client.post(
        "/solve", content=document,
        headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert ("body", "prefix") in {
        tuple(e["loc"]) for e in resp.json()["detail"]}

    # Duplicate key inside an executed segment object.
    document = (
        '{"n": 2, "L": 2, "fee_a": 0, "fee_b": 0, '
        '"costs": [{"a": 0, "b": 0}, {"a": 0, "b": 0}], '
        '"prefix": [{"start": 0, "start": 1, "end": 2, "source": "A"}]}'
    )
    resp = client.post(
        "/solve", content=document,
        headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert ("body", "prefix", 0, "start") in {
        tuple(e["loc"]) for e in resp.json()["detail"]}


def test_api_unknown_field_alongside_prefix_is_422():
    resp = client.post("/solve", json=_payload(
        prefix=[{"start": 0, "end": 1, "source": "A"}], extension=1))
    assert resp.status_code == 422
    assert ("body", "extension") in {
        tuple(e["loc"]) for e in resp.json()["detail"]}


def test_api_prefix_combines_with_windows_and_objectives():
    # End-to-end: a window forces the suffix source; both objectives agree on
    # cost/certainty and preserve the executed segment.
    payload = _payload(
        n=4, L=4, fee_a=0, fee_b=1_000_000,
        costs=[{"a": 0, "b": 1_000_000}] * 4,
        unavailable={"A": [{"start": 1, "end": 3}], "B": []},
        prefix=[{"start": 0, "end": 1, "source": "A"}],
        max_segments=3)
    default_resp = client.post("/solve", json=payload)
    cont_resp = client.post(
        "/solve", json={**payload, "objective": "continuity"})
    assert default_resp.status_code == cont_resp.status_code == 200
    d, c = default_resp.json(), cont_resp.json()
    assert d["segments"][0] == {"start": 0, "end": 1, "source": "A"}
    assert d["cost"] == c["cost"]
    assert d["certainty"] == c["certainty"]
    # The middle positions cannot be served by any A segment spanning the
    # window adjacent to the fixed prefix... they are covered by B here.
    assert all(seg["source"] == "B" for seg in d["segments"][1:-1]) \
        or all(seg["source"] == "B" for seg in d["segments"][1:])
    for seg in d["segments"]:
        assert 1 <= seg["end"] - seg["start"] <= 4


def test_openapi_documents_prefix():
    schema = app.openapi()
    gap = schema["components"]["schemas"]["GapRequest"]
    assert "prefix" in gap["properties"]
    assert "prefix" not in gap.get("required", [])
    assert "ExecutedSegment" in schema["components"]["schemas"]
    segment_schema = schema["components"]["schemas"]["ExecutedSegment"]
    assert set(segment_schema["properties"]) == {"start", "end", "source"}
    assert segment_schema.get("additionalProperties") is False
