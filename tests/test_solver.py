"""Acceptance tests.

* Small exhaustive oracles enumerate every legal cut and A/B source
  assignment for both objectives, including recursive tie keys.
* Independent O(nL) and lazy-heap references use the exact same tie
  breakers, including the continuity prefix-state recurrence.
* Every returned plan is replayed from the recurrence and must reproduce
  the reported optimum exactly.
* Malformed length, out-of-range values, illegal L and illegal objectives
  give 422 and never a partial plan.
"""

import random
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import SOURCE_A, SOURCE_B, Segment, solve

client = TestClient(app)


# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------


def recompute(costs, fees, segments):
    """Sum segment costs directly from the definition."""
    fees = {"A": fees[0], "B": fees[1]}
    total = 0
    for start, end, source in segments:
        s = 0 if source == "A" else 1
        total += fees[source] + sum(costs[k][s] for k in range(start, end))
    return total


def oracle(n, costs, fee_a, fee_b, max_len):
    """Brute-force DP that enumerates every (i, source) for every endpoint.

    Comparison order is exactly (cost, segment count, predecessor, source)
    with source A < B, so it returns the same unique plan as the solver.
    """
    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    best = {0: (0, 0, -1, -1)}  # j -> (cost, count, prev, source)
    for j in range(1, n + 1):
        cand = None
        for i in range(max(0, j - max_len), j):
            for s in (SOURCE_A, SOURCE_B):
                c = (
                    best[i][0] + fees[s] + pref[s][j] - pref[s][i],
                    best[i][1] + 1,
                    i,
                    s,
                )
                if cand is None or c < cand:
                    cand = c
        best[j] = cand

    segs = []
    j = n
    while j > 0:
        _, _, i, s = best[j]
        segs.append((i, j, "AB"[s]))
        j = i
    segs.reverse()
    return best[n][0], segs


def enumerate_plans(n, max_len):
    """Enumerate every legal segmentation and A/B source assignment."""

    def dfs(start, plan):
        if start == n:
            yield tuple(plan)
            return
        upper = min(n, start + max_len)
        for end in range(start + 1, upper + 1):
            for source in (SOURCE_A, SOURCE_B):
                yield from dfs(end, plan + [(start, end, source)])

    yield from dfs(0, [])


def plan_cost(plan, pref, fees):
    total = 0
    for start, end, source in plan:
        total += fees[source] + pref[source][end] - pref[source][start]
    return total


def default_plan_key(plan, pref, fees):
    """Recursive default order, including the prefix's own order."""
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
    """Recursive continuity order, including the prefix's own order."""
    if not plan:
        return ()
    start, _end, source = plan[-1]
    prefix = plan[:-1]
    switches = sum(
        left[2] != right[2] for left, right in zip(plan, plan[1:])
    )
    return (
        plan_cost(plan, pref, fees),
        switches,
        len(plan),
        start,
        source,
        continuity_plan_key(prefix, pref, fees),
    )


def enumerated_oracle(n, costs, fee_a, fee_b, max_len, objective):
    """Choose directly from the complete set of feasible plans."""
    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    key_func = (
        continuity_plan_key if objective == "continuity"
        else default_plan_key
    )
    best_plan = min(
        enumerate_plans(n, max_len),
        key=lambda plan: key_func(plan, pref, fees),
    )
    return (
        plan_cost(best_plan, pref, fees),
        [(start, end, "AB"[source]) for start, end, source in best_plan],
    )


def certainty_oracle(n, costs, fee_a, fee_b, max_len):
    """Independent oracle for per-position source unions.

    Enumerates *every* legal segmentation and A/B source assignment, keeps
    those whose total cost equals the global minimum, and unions the sources
    covering each position. Tie-breaking (segment count, switches, ...) is
    deliberately ignored: every minimum-cost plan contributes.
    """
    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    optimum = min(plan_cost(plan, pref, fees)
                  for plan in enumerate_plans(n, max_len))
    possible = [[False, False] for _ in range(n)]
    for plan in enumerate_plans(n, max_len):
        if plan_cost(plan, pref, fees) != optimum:
            continue
        for start, end, source in plan:
            for k in range(start, end):
                possible[k][source] = True

    labels = []
    for a_possible, b_possible in possible:
        if a_possible and b_possible:
            labels.append("EITHER")
        elif a_possible:
            labels.append("A_ONLY")
        else:
            labels.append("B_ONLY")
    return labels


def continuity_reference(n, costs, fee_a, fee_b, max_len):
    """Independent O(nL) DP using the full recursive continuity ordering."""
    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    keys = [[None] * (n + 1) for _ in range(2)]
    previous = [[(-1, -1)] * (n + 1) for _ in range(2)]
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
                for prefix_state, prefix_keys in predecessor_options:
                    if i == 0:
                        if prefix_state != -1:
                            continue
                        candidate = (
                            fees[source] + pref[source][j],
                            0,
                            1,
                            i,
                            source,
                            (),
                        )
                    else:
                        if prefix_state == -1:
                            continue
                        prefix_key = prefix_keys[i]
                        (prefix_cost, prefix_switches,
                         prefix_count, _, _, _) = prefix_key
                        candidate = (
                            prefix_cost + fees[source]
                            + pref[source][j] - pref[source][i],
                            prefix_switches + (prefix_state != source),
                            prefix_count + 1,
                            i,
                            source,
                            prefix_key,
                        )
                    if best is None or candidate < best:
                        best = candidate
                        chosen = (i, prefix_state)

            keys[source][j] = best
            previous[source][j] = chosen

    final = min(keys[source][n] for source in range(2))
    state = final[4]
    end = n
    plan = []
    while end:
        start, next_state = previous[state][end]
        plan.append((start, end, "AB"[state]))
        end, state = start, next_state
    plan.reverse()
    return final[0], plan


def heap_reference(n, costs, fee_a, fee_b, max_len):
    """Independent O(n log n) reference using lazy-deletion heaps.

    Used for the full max-scale case where O(nL) is too slow.
    """
    import heapq

    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    costs_arr = [0] * (n + 1)
    counts = [0] * (n + 1)
    prev = [-1] * (n + 1)
    psrc = [-1] * (n + 1)

    # Heaps hold (val, count, index); expired/dead indices are skipped.
    heaps = [
        [(0, 0, 0)],  # source A
        [(0, 0, 0)],  # source B
    ]
    live_sets = [{0}, {0}]

    for j in range(1, n + 1):
        low = j - max_len
        candidates = []
        for s, heap in enumerate(heaps):
            live = live_sets[s]
            while heap:
                val, count, i = heap[0]
                if i < low or i not in live:
                    heapq.heappop(heap)
                else:
                    break
            val, count, i = heap[0]
            candidates.append((val + pref[s][j] + fees[s], count + 1, i, s))
        cost, count, i, s = min(candidates)
        costs_arr[j] = cost
        counts[j] = count
        prev[j] = i
        psrc[j] = s
        for t, heap in enumerate(heaps):
            entry = (costs_arr[j] - pref[t][j], counts[j], j)
            heapq.heappush(heap, entry)
            live_sets[t].add(j)

    segs = []
    j = n
    while j > 0:
        segs.append((prev[j], j, "AB"[psrc[j]]))
        j = prev[j]
    segs.reverse()
    return costs_arr[n], segs


def continuity_heap_reference(n, costs, fee_a, fee_b, max_len):
    """Independent lazy-heap reference for the continuity tie ordering."""
    import heapq

    fees = (fee_a, fee_b)
    pref = [[0] * (n + 1) for _ in range(2)]
    for k in range(n):
        pref[0][k + 1] = pref[0][k] + costs[k][0]
        pref[1][k + 1] = pref[1][k] + costs[k][1]

    state_cost = [[0] * (n + 1) for _ in range(2)]
    state_switches = [[0] * (n + 1) for _ in range(2)]
    state_counts = [[0] * (n + 1) for _ in range(2)]
    state_start = [[0] * (n + 1) for _ in range(2)]
    prev = [[-1] * (n + 1) for _ in range(2)]
    previous_state = [[-1] * (n + 1) for _ in range(2)]

    # Rows 0/1 are real previous-source states; row 2 is the no-source
    # sentinel that only creates first segments.
    sentinel_entry = (0, 0, 0, 0, ())
    heaps = [
        [[sentinel_entry], [sentinel_entry]],
        [[sentinel_entry], [sentinel_entry]],
        [[sentinel_entry], [sentinel_entry]],
    ]

    for j in range(1, n + 1):
        low = j - max_len
        for source in range(2):
            best = None
            chosen = None
            for p in range(3):
                heap = heaps[p][source]
                while heap and heap[0][3] < low:
                    heapq.heappop(heap)
                if not heap:
                    continue
                val, prefix_switches, prefix_count, i, prefix_key = heap[0]
                candidate = (
                    val + pref[source][j] + fees[source],
                    prefix_switches + (0 if p == source or p == 2 else 1),
                    prefix_count + 1,
                    i,
                    source,
                    prefix_key,
                )
                if best is None or candidate < best:
                    best = candidate
                    chosen = (i, p)

            cost, switch_count, count, i, source, _ = best
            state_cost[source][j] = cost
            state_switches[source][j] = switch_count
            state_counts[source][j] = count
            state_start[source][j] = i
            prev[source][j] = chosen[0]
            previous_state[source][j] = chosen[1] if chosen[1] < 2 else -1

            full_key = best
            for new_source, heap in enumerate(heaps[source]):
                heapq.heappush(
                    heap,
                    (
                        cost - pref[new_source][j],
                        switch_count,
                        count,
                        j,
                        full_key,
                    ),
                )

    final = min(
        (
            state_cost[p][n],
            state_switches[p][n],
            state_counts[p][n],
            state_start[p][n],
            p,
        )
        for p in range(2)
    )
    state = final[4]
    end = n
    plan = []
    while end:
        start = prev[state][end]
        plan.append((start, end, "AB"[state]))
        end, state = start, previous_state[state][end]
    plan.reverse()
    return final[0], plan


def random_instance(rng, n, L, vmax=10, fmax=10):
    costs = [(rng.randrange(0, vmax + 1), rng.randrange(0, vmax + 1))
             for _ in range(n)]
    return n, costs, rng.randrange(0, fmax + 1), rng.randrange(0, fmax + 1), L


def assert_plan_valid(plan, n, max_len, costs, fees, reported_cost):
    # Coverage: back-to-back half-open intervals spanning exactly [0, n).
    assert plan[0].start == 0
    assert plan[-1].end == n
    for a, b in zip(plan, plan[1:]):
        assert a.end == b.start
    for seg in plan:
        assert seg.source in ("A", "B")
        assert 1 <= seg.end - seg.start <= max_len
    # Exact replay from the cost definition.
    assert recompute(costs, fees,
                     [(s.start, s.end, s.source) for s in plan]) == reported_cost


# --------------------------------------------------------------------------
# Exhaustive oracle for n <= 8
# --------------------------------------------------------------------------


@pytest.mark.parametrize("objective", ["default", "continuity"])
@pytest.mark.parametrize("n", range(1, 9))
def test_matches_exhaustive_oracle_all_fees(n, objective):
    # Deterministic pseudo-random instances cover a useful variety of ties;
    # the oracle itself enumerates every legal cut/source combination.
    rng = random.Random(1000 + n)
    for trial in range(12):
        _, costs, fa, fb, L = random_instance(
            rng, n, rng.randrange(1, n + 1), vmax=5, fmax=6)
        sol = solve(n, costs, fa, fb, L, objective=objective)
        ocost, osegs = enumerated_oracle(
            n, costs, fa, fb, L, objective)
        assert sol.cost == ocost
        assert [(s.start, s.end, s.source) for s in sol.segments] == osegs
        assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)
        # Certainty is the union over ALL minimum-cost plans and is
        # independent of the objective's secondary adjudication.
        expected_certainty = certainty_oracle(n, costs, fa, fb, L)
        assert list(sol.certainty) == expected_certainty
        other = solve(n, costs, fa, fb, L,
                      objective="continuity" if objective == "default"
                      else "default")
        assert list(other.certainty) == expected_certainty


@pytest.mark.parametrize("objective", ["default", "continuity"])
def test_certainty_exhaustive_zero_costs(objective):
    # All-zero per-position costs with a variety of L: every cover costs the
    # same, so every position must be EITHER whenever a same-position source
    # choice exists. The independent enumerator is the authority.
    for n in range(1, 8):
        for L in range(1, n + 1):
            for fa, fb in ((0, 0), (0, 1), (2, 2)):
                costs = [(0, 0)] * n
                sol = solve(n, costs, fa, fb, L, objective=objective)
                assert list(sol.certainty) == \
                    certainty_oracle(n, costs, fa, fb, L)


def test_certainty_locked_case():
    # n=3, L=2, all fees zero, costs=[(0,5),(0,0),(5,0)]:
    # minimum cost 0. Position 0 can only be A (B there already costs 5);
    # position 2 only B; position 1 is served by either source across the
    # set of optimal covers.
    n, L = 3, 2
    costs = [(0, 5), (0, 0), (5, 0)]
    expected = ["A_ONLY", "EITHER", "B_ONLY"]
    default_sol = solve(n, costs, 0, 0, L)
    continuity_sol = solve(n, costs, 0, 0, L, objective="continuity")
    assert list(default_sol.certainty) == expected
    assert list(continuity_sol.certainty) == expected
    assert default_sol.cost == continuity_sol.cost == 0
    # Old fields still obey their own objective rules.
    assert [(s.start, s.end, s.source)
            for s in default_sol.segments] == [(0, 1, "A"), (1, 3, "B")]


def test_handcrafted_tie_breakers():
    # Zero fees and equal per-position costs: every cover costs the same.
    # Default plan must be: fewest segments, then smallest predecessor,
    # then source A.
    n, L = 5, 3
    costs = [(0, 0)] * n
    sol = solve(n, costs, 0, 0, L)
    # Fewest segments with length <= 3: a 2-segment cover. Cuts at 2
    # ([0,2)+[2,5)) and 3 ([0,3)+[3,5)) tie; the smaller predecessor
    # index wins, and source A wins the A/B tie.
    assert sol.cost == 0
    assert [(s.start, s.end, s.source) for s in sol.segments] == [
        (0, 2, "A"), (2, 5, "A")]
    sol2 = solve(n, [(1, 2)] * n, 0, 0, L)
    ocost, osegs = oracle(n, [(1, 2)] * n, 0, 0, L)
    assert sol2.cost == ocost
    assert [(s.start, s.end, s.source) for s in sol2.segments] == osegs


def test_objective_lock_case():
    n, L = 3, 1
    costs = [(0, 0), (1, 0), (0, 1)]
    default_sol = solve(n, costs, 0, 0, L)
    continuity_sol = solve(n, costs, 0, 0, L, objective="continuity")
    assert [(s.start, s.end, s.source)
            for s in default_sol.segments] == [
        (0, 1, "A"), (1, 2, "B"), (2, 3, "A")]
    assert [(s.start, s.end, s.source)
            for s in continuity_sol.segments] == [
        (0, 1, "B"), (1, 2, "B"), (2, 3, "A")]
    assert default_sol.cost == continuity_sol.cost == 0


# --------------------------------------------------------------------------
# Larger / max-scale cases
# --------------------------------------------------------------------------


def test_random_medium_against_oracle():
    rng = random.Random(42)
    for _ in range(30):
        n = rng.randrange(11, 60)
        L = rng.randrange(1, min(n, 12) + 1)
        _, costs, fa, fb, _ = random_instance(rng, n, L, vmax=100, fmax=50)
        sol = solve(n, costs, fa, fb, L)
        ocost, osegs = oracle(n, costs, fa, fb, L)
        assert sol.cost == ocost
        assert [(s.start, s.end, s.source) for s in sol.segments] == osegs


def test_random_medium_continuity_against_reference():
    rng = random.Random(424)
    for _ in range(30):
        n = rng.randrange(11, 50)
        L = rng.randrange(1, min(n, 10) + 1)
        _, costs, fa, fb, _ = random_instance(rng, n, L, vmax=20, fmax=8)
        sol = solve(n, costs, fa, fb, L, objective="continuity")
        rc_cost, rc_segs = continuity_reference(n, costs, fa, fb, L)
        assert sol.cost == rc_cost
        assert [(s.start, s.end, s.source) for s in sol.segments] == rc_segs


def test_max_scale_full():
    n, L = 200_000, 4096
    rng = random.Random(7)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001)

    t0 = time.perf_counter()
    sol = solve(n, costs, fa, fb, L)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0  # O(n), nowhere near enumeration cost

    ref_cost, ref_segs = heap_reference(n, costs, fa, fb, L)
    assert sol.cost == ref_cost
    assert [(s.start, s.end, s.source) for s in sol.segments] == ref_segs
    assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)


def test_max_scale_continuity():
    n, L = 200_000, 4096
    rng = random.Random(70)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001)

    t0 = time.perf_counter()
    sol = solve(n, costs, fa, fb, L, objective="continuity")
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0

    ref_cost, ref_segs = continuity_heap_reference(n, costs, fa, fb, L)
    assert sol.cost == ref_cost
    assert [(s.start, s.end, s.source) for s in sol.segments] == ref_segs
    assert_plan_valid(sol.segments, n, L, costs, (fa, fb), sol.cost)


def test_max_scale_certainty_performance_and_consistency():
    # n=200000, L=4096: certainty must stay O(n) and both objectives report
    # identical labels. With random costs/fees ties are effectively absent,
    # so each position is forced to a single source; the reported winner's
    # segments must be consistent with the labels.
    n, L = 200_000, 4096
    rng = random.Random(700)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001)

    t0 = time.perf_counter()
    sol = solve(n, costs, fa, fb, L)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0
    assert len(sol.certainty) == n
    assert set(sol.certainty) <= {"A_ONLY", "B_ONLY", "EITHER"}

    cont = solve(n, costs, fa, fb, L, objective="continuity")
    assert list(cont.certainty) == list(sol.certainty)

    # Every segment of the (unique) optimum agrees with its labels, and no
    # A_ONLY position is served by B or vice versa.
    label_at = sol.certainty
    for seg in sol.segments:
        for k in range(seg.start, seg.end):
            assert label_at[k] in (
                "EITHER", seg.source + "_ONLY")

    # Structural check against the cost-only prefix/suffix split: the label
    # set is recomputed directly from a naive O(nL) DP at small L below.


def test_certainty_matches_naive_prefix_suffix_split():
    # Independent O(nL) recomputation of the prefix/suffix optimality test,
    # rather than replaying the solver's own windows.
    def naive_certainty(n, costs, fa, fb, max_len):
        pref = [[0] * (n + 1) for _ in range(2)]
        for k in range(n):
            pref[0][k + 1] = pref[0][k] + costs[k][0]
            pref[1][k + 1] = pref[1][k] + costs[k][1]
        fees = (fa, fb)
        inf = float("inf")
        f = [inf] * (n + 1)
        f[0] = 0
        for j in range(1, n + 1):
            for i in range(max(0, j - max_len), j):
                for s in (SOURCE_A, SOURCE_B):
                    v = f[i] + fees[s] + pref[s][j] - pref[s][i]
                    if v < f[j]:
                        f[j] = v
        g = [inf] * (n + 1)
        g[n] = 0
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, min(n, i + max_len) + 1):
                for s in (SOURCE_A, SOURCE_B):
                    v = fees[s] + pref[s][j] - pref[s][i] + g[j]
                    if v < g[i]:
                        g[i] = v
        optimum = f[n]
        poss = [[False, False] for _ in range(n)]
        for j in range(1, n + 1):
            for i in range(max(0, j - max_len), j):
                for s in (SOURCE_A, SOURCE_B):
                    if f[i] + fees[s] + pref[s][j] - pref[s][i] + g[j] \
                            == optimum:
                        for k in range(i, j):
                            poss[k][s] = True
        return ["EITHER" if a and b else "A_ONLY" if a else "B_ONLY"
                for a, b in poss]

    rng = random.Random(31337)
    # Small-L max-scale case keeps the O(nL) oracle feasible.
    n, L = 200_000, 3
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = 500_000, 500_000
    sol = solve(n, costs, fa, fb, L)
    assert list(sol.certainty) == naive_certainty(n, costs, fa, fb, L)

    # Plus a handful of smaller randomized cross-checks.
    rng = random.Random(31338)
    for _ in range(25):
        m = rng.randrange(1, 40)
        m_l = rng.randrange(1, min(m, 8) + 1)
        c = [(rng.randrange(0, 12), rng.randrange(0, 12)) for _ in range(m)]
        a_fee, b_fee = rng.randrange(0, 6), rng.randrange(0, 6)
        got = solve(m, c, a_fee, b_fee, m_l)
        assert list(got.certainty) == \
            naive_certainty(m, c, a_fee, b_fee, m_l)


def test_larger_than_n_window():
    # L may exceed n; the window simply covers every predecessor.
    rng = random.Random(2026)
    for n in (1, 2, 5, 10):
        _, costs, fa, fb, _ = random_instance(
            rng, n, 4096, vmax=1000, fmax=1000)
        sol = solve(n, costs, fa, fb, 4096)
        ocost, osegs = oracle(n, costs, fa, fb, 4096)
        assert sol.cost == ocost
        assert [(s.start, s.end, s.source) for s in sol.segments] == osegs
        assert_plan_valid(sol.segments, n, 4096, costs, (fa, fb), sol.cost)


def test_max_scale_with_small_L_matches_naive_dp():
    n, L = 200_000, 3
    rng = random.Random(11)
    costs = [(rng.randrange(0, 1_000_001), rng.randrange(0, 1_000_001))
             for _ in range(n)]
    fa, fb = 500_000, 500_000
    sol = solve(n, costs, fa, fb, L)
    ocost, osegs = oracle(n, costs, fa, fb, L)  # 6e5 ops, fine
    assert sol.cost == ocost
    assert [(s.start, s.end, s.source) for s in sol.segments] == osegs


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


def test_api_happy_path_and_replay():
    resp = client.post("/solve", json=_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cost", "segments", "certainty"}
    n = 5
    costs = [(k, k + 1) for k in range(n)]
    assert_plan_valid(
        [Segment(s["start"], s["end"], s["source"]) for s in data["segments"]],
        n, 3, costs, (1, 2), data["cost"])
    ocost, osegs = oracle(n, costs, 1, 2, 3)
    assert data["cost"] == ocost
    assert [(s["start"], s["end"], s["source"]) for s in
            data["segments"]] == osegs
    assert data["certainty"] == certainty_oracle(n, costs, 1, 2, 3)
    assert len(data["certainty"]) == n
    assert set(data["certainty"]) <= {"A_ONLY", "B_ONLY", "EITHER"}


def test_api_default_objective_compatibility():
    omitted = client.post("/solve", json=_payload())
    explicit = client.post("/solve", json=_payload(objective="default"))
    assert omitted.status_code == explicit.status_code == 200
    assert omitted.json() == explicit.json()


def test_api_certainty_objective_independent():
    # certainty describes the full set of minimum-cost covers, so it must be
    # byte-for-byte identical across objectives even when segments differ.
    payload = _payload(
        n=3,
        L=1,
        fee_a=0,
        fee_b=0,
        costs=[{"a": 0, "b": 0}, {"a": 1, "b": 0}, {"a": 0, "b": 1}],
    )
    default_resp = client.post("/solve", json=payload)
    continuity_resp = client.post(
        "/solve", json={**payload, "objective": "continuity"})
    assert default_resp.json()["certainty"] == \
        continuity_resp.json()["certainty"]
    assert default_resp.json()["cost"] == continuity_resp.json()["cost"]
    # Secondary adjudication may change the reported segments but never the
    # set of possible per-position sources.
    assert default_resp.json()["segments"] != \
        continuity_resp.json()["segments"]


def test_api_continuity_objective():
    payload = _payload(
        n=3,
        L=1,
        fee_a=0,
        fee_b=0,
        costs=[{"a": 0, "b": 0}, {"a": 1, "b": 0}, {"a": 0, "b": 1}],
    )
    default_resp = client.post("/solve", json=payload)
    continuity_resp = client.post(
        "/solve", json={**payload, "objective": "continuity"})
    assert default_resp.status_code == continuity_resp.status_code == 200
    assert default_resp.json()["segments"] == [
        {"start": 0, "end": 1, "source": "A"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
    assert continuity_resp.json()["segments"] == [
        {"start": 0, "end": 1, "source": "B"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
    assert set(continuity_resp.json()) == {"cost", "segments", "certainty"}
    assert continuity_resp.json()["certainty"] == \
        default_resp.json()["certainty"]


def test_api_max_scale():
    n = 200_000
    rng = random.Random(99)
    costs = [{"a": rng.randrange(1_000_001), "b": rng.randrange(1_000_001)}
             for _ in range(n)]
    resp = client.post("/solve", json={"n": n, "L": 4096,
                                       "fee_a": 10, "fee_b": 20,
                                       "costs": costs})
    assert resp.status_code == 200
    data = resp.json()
    segs = data["segments"]
    assert segs[0]["start"] == 0 and segs[-1]["end"] == n
    assert all(b["start"] == a["end"] for a, b in zip(segs, segs[1:]))
    assert all(1 <= s["end"] - s["start"] <= 4096 for s in segs)
    assert len(data["certainty"]) == n
    assert set(data["certainty"]) <= {"A_ONLY", "B_ONLY", "EITHER"}
    # Labels agree with the reported (unique-optimum) segmentation.
    for seg in segs:
        for k in range(seg["start"], seg["end"]):
            assert data["certainty"][k] in (
                "EITHER", seg["source"] + "_ONLY")


@pytest.mark.parametrize("payload", [
    # length mismatch: costs shorter / longer than n
    lambda: _payload(n=5, costs=[{"a": 1, "b": 1}] * 4),
    lambda: _payload(n=5, costs=[{"a": 1, "b": 1}] * 6),
    # value out of range
    lambda: _payload(fee_a=-1),
    lambda: _payload(fee_b=1_000_001),
    lambda: _payload(costs=[{"a": -1, "b": 0}] + [{"a": 0, "b": 0}] * 4),
    lambda: _payload(costs=[{"a": 0, "b": 1_000_001}] + [{"a": 0, "b": 0}] * 4),
    # n out of range
    lambda: _payload(n=0, costs=[]),
    lambda: _payload(n=200_001, costs=[{"a": 0, "b": 0}] * 200_001),
    # illegal L
    lambda: _payload(L=0),
    lambda: _payload(L=4097),
    # illegal objective
    lambda: _payload(objective=""),
    lambda: _payload(objective="CONTINUITY"),
    lambda: _payload(objective="lowest-switch"),
    lambda: {**_payload(), "objective": None},
    lambda: {**_payload(), "objective": 1},
    # wrong types
    lambda: {**_payload(), "n": "5"},
    lambda: {**_payload(), "L": True},
    lambda: _payload(costs=[{"a": 1.5, "b": 0}] + [{"a": 0, "b": 0}] * 4),
    lambda: _payload(costs=[{"a": "1", "b": 0}] + [{"a": 0, "b": 0}] * 4),
    # missing field
    lambda: {k: v for k, v in _payload().items() if k != "fee_b"},
])
def test_api_validation_errors_422(payload):
    resp = client.post("/solve", json=payload())
    assert resp.status_code == 422
    body = resp.json()
    # Only the standard validation payload; never a partial plan.
    assert set(body) == {"detail"}
    assert "cost" not in body and "segments" not in body
    assert "certainty" not in body


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
