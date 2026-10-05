"""Tests for optional per-source maintenance windows (``unavailable``).

Short sequences are checked against oracles that enumerate *every* legal
segmentation and A/B source assignment after deleting segments intersecting a
source window -- including full recursive tie keys for both objectives and
the certainty union over every minimum-cost legal plan. Independent O(nL)
references cross-check medium/random instances and the maximum-scale case
uses a small L so the naive prefix/suffix split stays affordable.

Coverage:
* half-open window endpoints (start usable, end usable, interior blocked);
* a single source available for the whole gap;
* both sources down at the same position (feasibility and 409);
* touching / overlapping / unsorted windows rejected with detail-only 422;
* omitted / null / empty ``unavailable`` byte-for-byte legacy regression.
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
    solve,
)

client = TestClient(app)


# --------------------------------------------------------------------------
# Availability-aware enumeration oracles
# --------------------------------------------------------------------------


def windows_to_blocked(n, windows_by_source):
    """Return a per-source frozenset of blocked positions."""
    blocked = []
    for windows in windows_by_source:
        positions = set()
        for start, end in windows:
            positions.update(range(start, end))
        blocked.append(frozenset(positions))
    return tuple(blocked)


def enumerate_plans_avail(n, max_len, blocked):
    """Enumerate every segmentation/source plan avoiding blocked positions."""

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
    total = 0
    for start, end, source in plan:
        total += fees[source] + pref[source][end] - pref[source][start]
    return total


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


def oracle_all(n, costs, fa, fb, max_len, windows_by_source):
    """Full enumeration oracle.

    Returns ``None`` when no legal cover exists, otherwise
    ``(optimum, default_segments, continuity_segments, certainty)``.
    """
    fees = (fa, fb)
    pref = prefix_sums(n, costs)
    blocked = windows_to_blocked(n, windows_by_source)
    plans = list(enumerate_plans_avail(n, max_len, blocked))
    if not plans:
        return None

    optimum = min(plan_cost(plan, pref, fees) for plan in plans)
    optimal = [p for p in plans if plan_cost(p, pref, fees) == optimum]

    default_best = min(optimal, key=lambda p: default_plan_key(p, pref, fees))
    continuity_best = min(
        optimal, key=lambda p: continuity_plan_key(p, pref, fees))

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


def all_window_configs(n):
    """Every pair of per-source blocked subsets as canonical window lists.

    Returns ``(windows_a, windows_b)`` pairs built from maximal runs of
    blocked positions; the solver contract forbids touching windows, which
    is exactly the maximal-run representation.
    """

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


def assert_segments_avoid_windows(segments, windows_by_source):
    blocked = windows_to_blocked(
        max(end for seg in segments for end in (seg.end,)) or 1,
        windows_by_source,
    )
    for seg in segments:
        s = SOURCE_A if seg.source == "A" else SOURCE_B
        for k in range(seg.start, seg.end):
            assert k not in blocked[s]


def replay_cost(segments, costs, fa, fb):
    fees = {"A": fa, "B": fb}
    total = 0
    for seg in segments:
        total += fees[seg.source] + sum(
            costs[k][0 if seg.source == "A" else 1]
            for k in range(seg.start, seg.end)
        )
    return total


# --------------------------------------------------------------------------
# Independent O(nL) blocked-aware references for medium instances
# --------------------------------------------------------------------------


def block_prefix(n, windows_by_source):
    pref = [[0] * (n + 1) for _ in range(2)]
    blocked = windows_to_blocked(n, windows_by_source)
    for s in (SOURCE_A, SOURCE_B):
        for k in range(n):
            pref[s][k + 1] = pref[s][k] + (k in blocked[s])
    return pref


def segment_ok(block_pref, s, i, j):
    return block_pref[s][j] - block_pref[s][i] == 0


def naive_optimum(n, costs, fa, fb, max_len, windows_by_source):
    """O(nL) cost-only DP; returns None when the gap cannot be covered."""
    inf = float("inf")
    fees = (fa, fb)
    pref = prefix_sums(n, costs)
    bp = block_prefix(n, windows_by_source)
    f = [inf] * (n + 1)
    f[0] = 0
    for j in range(1, n + 1):
        for i in range(max(0, j - max_len), j):
            if f[i] == inf:
                continue
            for s in (SOURCE_A, SOURCE_B):
                if not segment_ok(bp, s, i, j):
                    continue
                v = f[i] + fees[s] + pref[s][j] - pref[s][i]
                if v < f[j]:
                    f[j] = v
    return None if f[n] == inf else f[n]


def naive_certainty(n, costs, fa, fb, max_len, windows_by_source):
    """O(nL) prefix/suffix split oracle for labels under availability."""
    inf = float("inf")
    fees = (fa, fb)
    pref = prefix_sums(n, costs)
    bp = block_prefix(n, windows_by_source)

    f = [inf] * (n + 1)
    f[0] = 0
    for j in range(1, n + 1):
        for i in range(max(0, j - max_len), j):
            if f[i] == inf:
                continue
            for s in (SOURCE_A, SOURCE_B):
                if not segment_ok(bp, s, i, j):
                    continue
                v = f[i] + fees[s] + pref[s][j] - pref[s][i]
                if v < f[j]:
                    f[j] = v

    g = [inf] * (n + 1)
    g[n] = 0
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, min(n, i + max_len) + 1):
            if g[j] == inf:
                continue
            for s in (SOURCE_A, SOURCE_B):
                if not segment_ok(bp, s, i, j):
                    continue
                v = fees[s] + pref[s][j] - pref[s][i] + g[j]
                if v < g[i]:
                    g[i] = v

    if f[n] == inf:
        return None
    optimum = f[n]
    poss = [[False, False] for _ in range(n)]
    for j in range(1, n + 1):
        if g[j] == inf:
            continue
        for i in range(max(0, j - max_len), j):
            if f[i] == inf:
                continue
            for s in (SOURCE_A, SOURCE_B):
                if not segment_ok(bp, s, i, j):
                    continue
                if f[i] + fees[s] + pref[s][j] - pref[s][i] + g[j] \
                        == optimum:
                    for k in range(i, j):
                        poss[k][s] = True
    return tuple(
        "EITHER" if a and b else "A_ONLY" if a else "B_ONLY"
        for a, b in poss
    )


def continuity_naive_reference(n, costs, fa, fb, max_len,
                               windows_by_source):
    """O(nL) continuity DP over full recursive keys, availability-aware."""
    fees = (fa, fb)
    pref = prefix_sums(n, costs)
    bp = block_prefix(n, windows_by_source)

    keys = [[None] * (n + 1) for _ in range(2)]
    previous = [[None] * (n + 1) for _ in range(2)]
    predecessor_options = (
        (-1, None),
        (0, keys[0]),
        (1, keys[1]),
    )

    for j in range(1, n + 1):
        for source in range(2):
            best = None
            chosen = None
            low = max(0, j - max_len)
            for i in range(low, j):
                if not segment_ok(bp, source, i, j):
                    continue
                for prefix_state, prefix_keys in predecessor_options:
                    if i == 0:
                        if prefix_state != -1:
                            continue
                        candidate = (
                            fees[source] + pref[source][j], 0, 1,
                            i, source, (),
                        )
                    else:
                        if prefix_state == -1 or prefix_keys[i] is None:
                            continue
                        prefix_key = prefix_keys[i]
                        candidate = (
                            prefix_key[0] + fees[source]
                            + pref[source][j] - pref[source][i],
                            prefix_key[1] + (prefix_state != source),
                            prefix_key[2] + 1,
                            i, source, prefix_key,
                        )
                    if best is None or candidate < best:
                        best = candidate
                        chosen = (i, prefix_state)
            keys[source][j] = best
            previous[source][j] = chosen

    finals = [(keys[source][n], source) for source in range(2)
              if keys[source][n] is not None]
    if not finals:
        return None
    final, state = min(finals, key=lambda pair: pair[0])
    end = n
    plan = []
    while end:
        start, next_state = previous[state][end]
        plan.append((start, end, "AB"[state]))
        end, state = start, next_state
    plan.reverse()
    return final[0], plan


# --------------------------------------------------------------------------
# Exhaustive cross-check over every blocked-subset pair
# --------------------------------------------------------------------------


PROFILES = (
    ("zero", lambda k: (0, 0), 0, 0),
    ("fees", lambda k: (0, 0), 2, 3),
)


@pytest.mark.parametrize("objective", ["default", "continuity"])
@pytest.mark.parametrize("n", range(1, 6))
def test_exhaustive_all_window_subsets(n, objective):
    # Deterministic pseudo-random cost profile in addition to the tie-heavy
    # zero/fee profiles above.
    rng = random.Random(9000 + n)
    cost_profiles = [
        ([(0, 0)] * n, 0, 0),
        ([(0, 0)] * n, 2, 3),
        ([(rng.randrange(0, 4), rng.randrange(0, 4)) for _ in range(n)],
         rng.randrange(0, 4), rng.randrange(0, 4)),
    ]
    for L in range(1, n + 1):
        for costs, fa, fb in cost_profiles:
            for windows in all_window_configs(n):
                expected = oracle_all(n, costs, fa, fb, L, windows)
                if expected is None:
                    with pytest.raises(NoFeasibleRepair):
                        solve(n, costs, fa, fb, L,
                              objective=objective, unavailable=windows)
                    continue
                ocost, odef, ocont, ocert = expected
                osegs = odef if objective == "default" else ocont
                sol = solve(n, costs, fa, fb, L,
                            objective=objective, unavailable=windows)
                assert sol.cost == ocost
                got = [(s.start, s.end, s.source) for s in sol.segments]
                assert got == osegs
                assert tuple(sol.certainty) == ocert
                # Structural legality: back-to-back, bounded length, and no
                # segment intersects its own source's windows.
                assert sol.segments[0].start == 0
                assert sol.segments[-1].end == n
                for a, b in zip(sol.segments, sol.segments[1:]):
                    assert a.end == b.start
                for seg in sol.segments:
                    assert 1 <= seg.end - seg.start <= L
                assert_segments_avoid_windows(sol.segments, windows)
                assert replay_cost(sol.segments, costs, fa, fb) == ocost
                # certainty is objective-independent under availability too.
                other = solve(n, costs, fa, fb, L,
                              objective="continuity" if objective == "default"
                              else "default", unavailable=windows)
                assert tuple(other.certainty) == ocert


def test_exhaustive_certainty_matches_enumeration():
    # Dedicated certainty sweep over tie-heavy instances where many plans
    # share the optimum; the labels must equal the union over ALL optimal
    # legal plans, not just the adjudicated one.
    rng = random.Random(4242)
    for n in range(1, 7):
        costs = [(rng.randrange(0, 3), rng.randrange(0, 3))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 3), rng.randrange(0, 3)
        for L in range(1, n + 1):
            for windows in all_window_configs(n):
                expected = oracle_all(n, costs, fa, fb, L, windows)
                if expected is None:
                    continue
                _, _, _, ocert = expected
                for objective in ("default", "continuity"):
                    sol = solve(n, costs, fa, fb, L,
                                objective=objective, unavailable=windows)
                    assert tuple(sol.certainty) == ocert


# --------------------------------------------------------------------------
# Locked endpoint / single-source / simultaneous-down cases
# --------------------------------------------------------------------------


def test_half_open_window_endpoints():
    # n=2, L=2, fees 0. A unavailable on [0,1): position 0 blocked for A but
    # position 1 is usable; the half-open end is NOT blocked.
    n, L, costs, fa, fb = 2, 2, [(0, 0), (0, 0)], 0, 0
    sol = solve(n, costs, fa, fb, L, unavailable=([(0, 1)], []))
    got = [(s.start, s.end, s.source) for s in sol.segments]
    # A[1,2) legal, but pos 0 needs B; cheapest adjudication gives B[0,2)?
    # B[0,2) one segment ties B[0,1)+A[1,2) on cost, fewer segments wins.
    expected = oracle_all(n, costs, fa, fb, L, ([(0, 1)], []))
    assert sol.cost == expected[0]
    assert got == expected[1]
    assert tuple(sol.certainty) == expected[3]
    # Explicit half-open guarantees: an A segment may start exactly at end.
    assert any(s.source == "A" and s.start == 1 for s in sol.segments) or \
        all(s.source == "B" for s in sol.segments)

    # Window ending exactly at n: [n-1, n) blocks only the last position.
    sol2 = solve(n, costs, fa, fb, L, unavailable=([(1, 2)], []))
    expected2 = oracle_all(n, costs, fa, fb, L, ([(1, 2)], []))
    assert [(s.start, s.end, s.source) for s in sol2.segments] == expected2[1]
    assert tuple(sol2.certainty) == expected2[3]
    for seg in sol2.segments:
        if seg.source == "A":
            assert seg.end <= 1


def test_window_endpoint_equal_n_allowed():
    # end == n must validate; the solver must route around the tail window.
    n, L = 4, 4
    costs = [(1, 100)] * n
    fa = fb = 0
    # A down on the last position: the A run must stop at n-1 and B closes.
    sol = solve(n, costs, fa, fb, L, unavailable=([(n - 1, n)], []))
    assert_segments_avoid_windows(sol.segments, ([(n - 1, n)], []))
    assert sol.segments[-1].source == "B"
    assert sol.segments[-1].start == n - 1
    expected = oracle_all(n, costs, fa, fb, L, ([(n - 1, n)], []))
    assert sol.cost == expected[0]
    assert tuple(sol.certainty) == expected[3]


def test_single_source_available():
    # B under maintenance for the whole gap: every segment must be A.
    n, L = 5, 2
    costs = [(k, 1000) for k in range(n)]
    sol = solve(n, costs, 0, 0, L, unavailable=([], [(0, n)]))
    assert {s.source for s in sol.segments} == {"A"}
    expected = oracle_all(n, costs, 0, 0, L, ([], [(0, n)]))
    assert sol.cost == expected[0]
    assert [(s.start, s.end, s.source) for s in sol.segments] == expected[1]
    assert tuple(sol.certainty) == ("A_ONLY",) * n

    # A fully down and L=1: feasible only with B singleton segments.
    sol = solve(n, [(0, k) for k in range(n)], 1, 1, 1,
                unavailable=([(0, n)], []))
    assert {s.source for s in sol.segments} == {"B"}
    assert all(s.end - s.start == 1 for s in sol.segments)
    assert sol.cost == n + sum(range(n))


def test_both_sources_down_simultaneously():
    # n=3, L=2, both sources down on the middle position: no segment may
    # cross it and no segment can cover it -> infeasible.
    n, L = 3, 2
    costs = [(0, 0)] * n
    with pytest.raises(NoFeasibleRepair):
        solve(n, costs, 0, 0, L, unavailable=([(1, 2)], [(1, 2)]))

    # L=1 any simultaneous outage is immediately infeasible.
    with pytest.raises(NoFeasibleRepair):
        solve(4, costs + [(0, 0)], 0, 0, 1,
              unavailable=([(0, 1), (2, 4)], [(1, 3)]))


def test_unreachable_gap_between_windows():
    # A down on {0,2}, B down on {1}, n=3, L=1: the unique legal chain is
    # B,A,B. Removing B at 2 as well leaves position 2 uncovered.
    n = 3
    costs = [(0, 0)] * n
    sol = solve(n, costs, 0, 0, 1,
                unavailable=([(0, 1), (2, 3)], [(1, 2)]))
    assert [(s.start, s.end, s.source) for s in sol.segments] == [
        (0, 1, "B"), (1, 2, "A"), (2, 3, "B")]
    with pytest.raises(NoFeasibleRepair):
        solve(n, costs, 0, 0, 1,
                unavailable=([(0, 1), (2, 3)], [(1, 3)]))


def test_multiple_windows_same_source_forces_splits():
    # n=6, L=6, fees positive; A alternates down on {1,3,5}. A legal A plan
    # would need short segments; the oracle decides the adjudication.
    n, L = 6, 6
    costs = [(k % 2, 1 - k % 2) for k in range(n)]
    windows = ([(1, 2), (3, 4), (5, 6)], [])
    for objective in ("default", "continuity"):
        sol = solve(n, costs, 4, 4, L,
                    objective=objective, unavailable=windows)
        expected = oracle_all(n, costs, 4, 4, L, windows)
        osegs = expected[1] if objective == "default" else expected[2]
        assert sol.cost == expected[0]
        assert [(s.start, s.end, s.source) for s in sol.segments] == osegs
        assert tuple(sol.certainty) == expected[3]


def test_empty_unavailable_equivalent_to_omitted():
    # Solver-level regression: omitted, None, empty lists and empty tuples
    # must produce identical cost/segments/certainty on both objectives.
    rng = random.Random(555)
    for _ in range(60):
        n = rng.randrange(1, 30)
        L = rng.randrange(1, n + 2)
        costs = [(rng.randrange(0, 9), rng.randrange(0, 9))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 9), rng.randrange(0, 9)
        for objective in ("default", "continuity"):
            base = solve(n, costs, fa, fb, L, objective=objective)
            for empty in (None, ([], []), ((), ())):
                sol = solve(n, costs, fa, fb, L, objective=objective,
                            unavailable=empty)
                assert sol.cost == base.cost
                assert [(s.start, s.end, s.source) for s in sol.segments] == \
                    [(s.start, s.end, s.source) for s in base.segments]
                assert tuple(sol.certainty) == tuple(base.certainty)


# --------------------------------------------------------------------------
# Random medium instances against O(nL) references
# --------------------------------------------------------------------------


def random_windows(rng, n, p=0.25):
    result = []
    for _source in (SOURCE_A, SOURCE_B):
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


def test_random_medium_against_naive_references():
    rng = random.Random(7777)
    for trial in range(120):
        n = rng.randrange(1, 41)
        L = rng.randrange(1, min(n, 9) + 1)
        costs = [(rng.randrange(0, 12), rng.randrange(0, 12))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 7), rng.randrange(0, 7)
        windows = random_windows(rng, n)

        opt = naive_optimum(n, costs, fa, fb, L, windows)
        cert = naive_certainty(n, costs, fa, fb, L, windows)
        if opt is None:
            for objective in ("default", "continuity"):
                with pytest.raises(NoFeasibleRepair):
                    solve(n, costs, fa, fb, L,
                          objective=objective, unavailable=windows)
            continue

        sol = solve(n, costs, fa, fb, L, unavailable=windows)
        cont = solve(n, costs, fa, fb, L, objective="continuity",
                     unavailable=windows)
        assert sol.cost == cont.cost == opt
        assert tuple(sol.certainty) == tuple(cert)
        assert tuple(cont.certainty) == tuple(cert)
        assert_segments_avoid_windows(sol.segments, windows)
        assert_segments_avoid_windows(cont.segments, windows)
        assert replay_cost(sol.segments, costs, fa, fb) == opt
        assert replay_cost(cont.segments, costs, fa, fb) == opt

        rc = continuity_naive_reference(n, costs, fa, fb, L, windows)
        assert rc is not None
        rc_cost, rc_segs = rc
        assert rc_cost == opt
        assert [(s.start, s.end, s.source) for s in cont.segments] == \
            [(s, e, src) for s, e, src in rc_segs]


def test_random_windows_against_full_enumeration():
    # Random window patterns on short sequences still cross-checked against
    # the complete enumeration oracle for both objectives at once.
    rng = random.Random(313373)
    for _ in range(300):
        n = rng.randrange(1, 9)
        L = rng.randrange(1, n + 1)
        costs = [(rng.randrange(0, 5), rng.randrange(0, 5))
                 for _ in range(n)]
        fa, fb = rng.randrange(0, 5), rng.randrange(0, 5)
        windows = random_windows(rng, n, p=0.3)
        expected = oracle_all(n, costs, fa, fb, L, windows)
        if expected is None:
            for objective in ("default", "continuity"):
                with pytest.raises(NoFeasibleRepair):
                    solve(n, costs, fa, fb, L,
                          objective=objective, unavailable=windows)
            continue
        ocost, odef, ocont, ocert = expected
        d = solve(n, costs, fa, fb, L, unavailable=windows)
        c = solve(n, costs, fa, fb, L, objective="continuity",
                  unavailable=windows)
        assert d.cost == c.cost == ocost
        assert [(s.start, s.end, s.source) for s in d.segments] == odef
        assert [(s.start, s.end, s.source) for s in c.segments] == ocont
        assert tuple(d.certainty) == tuple(c.certainty) == ocert


# --------------------------------------------------------------------------
# Max-scale: availability stays O(n)
# --------------------------------------------------------------------------


def test_max_scale_with_windows_performance():
    import time

    n, L = 200_000, 4096
    rng = random.Random(20260)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001)
    # Sparse, independent outages (1%); long runs of availability remain.
    windows = random_windows(rng, n, p=0.01)

    t0 = time.perf_counter()
    sol = solve(n, costs, fa, fb, L, unavailable=windows)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0

    cont = solve(n, costs, fa, fb, L, objective="continuity",
                 unavailable=windows)
    assert sol.cost == cont.cost
    assert tuple(sol.certainty) == tuple(cont.certainty)
    assert_segments_avoid_windows(sol.segments, windows)
    assert_segments_avoid_windows(cont.segments, windows)
    assert sol.segments[0].start == 0 and sol.segments[-1].end == n
    assert all(1 <= s.end - s.start <= L for s in sol.segments)
    for seg in sol.segments:
        for k in range(seg.start, seg.end):
            assert sol.certainty[k] in ("EITHER", seg.source + "_ONLY")


def test_max_scale_small_L_naive_certainty_split():
    # n=200k, L=3 keeps the O(nL) blocked-aware oracle affordable.
    n, L = 200_000, 3
    rng = random.Random(20261)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = 500_000, 500_000
    # Low outage density keeps a feasible cover overwhelmingly likely;
    # regenerate the rare draw where a simultaneous outage blocks the gap.
    windows = None
    for _ in range(100):
        candidate = random_windows(rng, n, p=0.02)
        if naive_optimum(n, costs, fa, fb, L, candidate) is not None:
            windows = candidate
            break
    assert windows is not None

    sol = solve(n, costs, fa, fb, L, unavailable=windows)
    assert tuple(sol.certainty) == tuple(
        naive_certainty(n, costs, fa, fb, L, windows))
    assert_segments_avoid_windows(sol.segments, windows)
    cont = solve(n, costs, fa, fb, L, objective="continuity",
                 unavailable=windows)
    assert tuple(cont.certainty) == tuple(sol.certainty)


# --------------------------------------------------------------------------
# API behaviour
# --------------------------------------------------------------------------


def _payload(n=5, **overrides):
    p = {
        "n": n,
        "L": 3,
        "fee_a": 1,
        "fee_b": 2,
        "costs": [{"a": k, "b": k + 1} for k in range(n)],
    }
    p.update(overrides)
    return p


def test_api_unavailable_happy_path():
    n, L = 4, 2
    payload = _payload(
        n=n, L=L, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}] * n,
        unavailable={"A": [{"start": 1, "end": 2}],
                     "B": [{"start": 2, "end": 3}]},
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cost", "segments", "certainty"}
    segs = [(s["start"], s["end"], s["source"]) for s in data["segments"]]
    for start, end, source in segs:
        if source == "A":
            assert not (start <= 1 < end)
        else:
            assert not (start <= 2 < end)
    expected = oracle_all(
        n, [(0, 0)] * n, 0, 0, L, ([(1, 2)], [(2, 3)]))
    assert data["cost"] == expected[0]
    assert segs == expected[1]
    assert data["certainty"] == list(expected[3])


def test_api_409_body_is_detail_only():
    # Both sources down simultaneously at a position => 409 with exactly the
    # sentinel detail and no leaked local fields.
    payload = _payload(
        n=3, L=2,
        unavailable={"A": [{"start": 1, "end": 2}],
                     "B": [{"start": 1, "end": 2}]},
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 409
    assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}

    # A source outage longer than L spanning the gap, other source fully
    # down as well in a second instance.
    resp = client.post("/solve", json=_payload(
        n=5, L=2,
        unavailable={"A": [{"start": 0, "end": 5}],
                     "B": [{"start": 0, "end": 5}]}))
    assert resp.status_code == 409
    assert resp.json() == {"detail": "NO_FEASIBLE_REPAIR"}


def test_api_single_source_available():
    payload = _payload(
        n=4, L=2, fee_a=0, fee_b=10,
        costs=[{"a": 1, "b": 1}] * 4,
        unavailable={"A": [], "B": [{"start": 0, "end": 4}]},
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    assert all(s["source"] == "A" for s in resp.json()["segments"])
    assert set(resp.json()["certainty"]) <= {"A_ONLY"}


def test_api_window_endpoint_half_open():
    # A [0,1): position 0 blocked for A, position 1 free; response segments
    # must place A only at start >= 1.
    payload = _payload(
        n=2, L=2, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}, {"a": 0, "b": 0}],
        unavailable={"A": [{"start": 0, "end": 1}], "B": []},
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    for seg in resp.json()["segments"]:
        if seg["source"] == "A":
            assert seg["start"] >= 1


def test_api_omitted_null_empty_compatibility():
    base = _payload()
    omitted = client.post("/solve", json=base)
    nulled = client.post("/solve", json={**base, "unavailable": None})
    emptied = client.post(
        "/solve", json={**base, "unavailable": {"A": [], "B": []}})
    only_a = client.post(
        "/solve", json={**base, "unavailable": {"A": []}})
    assert omitted.status_code == nulled.status_code == \
        emptied.status_code == only_a.status_code == 200
    body = omitted.json()
    assert nulled.json() == body
    assert emptied.json() == body
    assert only_a.json() == body
    assert set(body) == {"cost", "segments", "certainty"}


def test_api_old_requests_regression():
    # Legacy payloads without the new field stay byte-for-byte stable.
    rng = random.Random(1234)
    for _ in range(10):
        n = rng.randrange(1, 12)
        payload = _payload(
            n=n, L=rng.randrange(1, n + 1),
            fee_a=rng.randrange(0, 20), fee_b=rng.randrange(0, 20),
            costs=[{"a": rng.randrange(0, 20), "b": rng.randrange(0, 20)}
                   for _ in range(n)],
            objective=rng.choice(["default", "continuity"]),
        )
        resp = client.post("/solve", json=payload)
        assert resp.status_code == 200
        assert "unavailable" not in payload
        assert set(resp.json()) == {"cost", "segments", "certainty"}


@pytest.mark.parametrize("unavailable", [
    # start >= end
    {"A": [{"start": 1, "end": 1}], "B": []},
    {"A": [], "B": [{"start": 3, "end": 2}]},
    # out of [0, n]
    {"A": [{"start": -1, "end": 2}], "B": []},
    {"A": [{"start": 0, "end": 6}], "B": []},
    {"A": [{"start": 5, "end": 5}], "B": []},
    {"A": [{"start": 6, "end": 7}], "B": []},
    {"A": [{"start": -2, "end": -1}], "B": []},
    # touching windows are rejected
    {"A": [{"start": 0, "end": 1}, {"start": 1, "end": 2}], "B": []},
    {"A": [], "B": [{"start": 2, "end": 3}, {"start": 3, "end": 4}]},
    # overlapping windows
    {"A": [{"start": 0, "end": 3}, {"start": 2, "end": 4}], "B": []},
    {"A": [{"start": 0, "end": 2}], "B": [
        {"start": 1, "end": 4}, {"start": 2, "end": 3}]},
    # not strictly increasing
    {"A": [{"start": 2, "end": 3}, {"start": 0, "end": 1}], "B": []},
    {"A": [{"start": 0, "end": 4}, {"start": 1, "end": 2}], "B": []},
    # wrong types
    [{"start": 0, "end": 1}],
    {"A": [{"start": 0, "end": "2"}], "B": []},
    {"A": [{"start": 0.5, "end": 1}], "B": []},
    {"A": [{"start": 0}], "B": []},
    {"C": []},
    {"A": [], "B": [], "C": [{"start": 0, "end": 1}]},
    5,
    "down",
])
def test_api_invalid_unavailable_422_detail_only(unavailable):
    resp = client.post("/solve", json=_payload(unavailable=unavailable))
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"detail"}
    assert "cost" not in body and "segments" not in body
    assert "certainty" not in body


def test_api_touching_windows_across_sources_allowed():
    # The no-touch rule is per source; A ending where B starts (or even
    # identical windows across sources) is legal and adjudicated normally.
    payload = _payload(
        n=3, L=1, fee_a=0, fee_b=0,
        costs=[{"a": 0, "b": 0}] * 3,
        unavailable={"A": [{"start": 0, "end": 1}],
                     "B": [{"start": 1, "end": 3}]},
    )
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200
    # L=1 forces singletons: pos 0 B, pos 1 A, pos 2 A.
    assert [(s["start"], s["end"], s["source"])
            for s in resp.json()["segments"]] == [
        (0, 1, "B"), (1, 2, "A"), (2, 3, "A")]


def test_api_both_objectives_share_certainty_under_windows():
    payload = _payload(
        n=4, L=3, fee_a=1, fee_b=1,
        costs=[{"a": k % 2, "b": 1 - k % 2} for k in range(4)],
        unavailable={"A": [{"start": 1, "end": 2}],
                     "B": [{"start": 2, "end": 3}]},
    )
    d = client.post("/solve", json=payload).json()
    c = client.post(
        "/solve", json={**payload, "objective": "continuity"}).json()
    assert d["cost"] == c["cost"]
    assert d["certainty"] == c["certainty"]
