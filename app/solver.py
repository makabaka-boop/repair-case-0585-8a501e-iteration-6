"""Sliding-window dynamic programs for satellite telemetry gap repair.

A gap of ``n`` positions ``[0, n)`` is covered by back-to-back half-open
segments ``[start, end)``. Every segment uses source A or B. Choosing source
``s`` for ``[i, j)`` costs

    fee[s] + sum(cost[k][s] for k in range(i, j))

so the activation fee is paid once per segment, including when adjacent
segments use the same source.

Each source may additionally be temporarily unavailable on a set of
half-open position windows (the optional ``unavailable`` argument, a pair of
``(windows_a, windows_b)``). A segment is legal only when it also lies
entirely outside its own source's windows. Availability is part of the
recurrences themselves -- predecessor ranges, final-source states, the
reverse minimum-cost sweep and the certainty merge -- never a post-hoc
filter on the unconstrained optimum. When no complete legal cover exists,
:class:`NoFeasibleRepair` is raised. With empty window lists every table
reduces to the plain ``[j-L, j-1]`` window, so old and new code paths are
the same computation.

The optional ``max_segments`` argument (request field ``max_segments``, an
integer 1..8) restricts the plan to at most that many segments. The segment
count then becomes part of the optimization state: every capped recurrence
is layered by the number of segments already used, and the certainty merge
keeps only minimum-cost plans that respect the cap.

Two distinct failure modes exist with a cap:

* availability alone makes the gap uncoverable -> :class:`NoFeasibleRepair`;
* a legal cover exists but every one needs more segments than the cap ->
  :class:`SegmentLimitExceeded`. The distinction is decided by a reachability
  pass that ignores cost, so "optimal uses more segments than allowed" can
  never be misreported as the ordinary failure.

Without ``max_segments`` the unbounded paths below run and every output is
bit-identical to the legacy computation.

The optional ``prefix`` argument fixes an already executed prefix: an ordered
tuple of ``(start, end, source_id)`` segments validated by the API layer to
tile ``[0, p)`` back-to-back, respect ``max_len`` and stay clear of their own
source's maintenance windows. When it is non-empty the executed segments are
kept verbatim but their cost is recomputed from the rates -- caller-supplied
fees are never accepted -- and every recurrence above is rerun as a *suffix*
problem seeded at ``p``:

* the seed carries the fixed cost, segment count, last source and the number
  of source switches inside the prefix;
* default/continuity adjudications then optimize only ``[p, n)``, with
  continuity counting the boundary switch from the prefix's last segment to
  the first new one;
* a segment cap subtracts the executed segment count (only ``[p, n)`` may
  still add segments), and certainty is decided solely over minimum-cost
  *completions* -- executed positions are forced to their fixed source while
  remaining positions reuse the prefix/suffix split seeded at ``p``;
* ``p == 0`` (empty/omitted prefix) runs the legacy paths unchanged; ``p ==
  n`` returns the fixed prefix repriced, with no suffix DP at all; a suffix
  that availability makes uncoverable is still ``NoFeasibleRepair`` and a
  remaining segment budget that cannot hold the minimum suffix cover is still
  :class:`SegmentLimitExceeded` -- no partial plan is ever returned.

The default objective compares candidates lexicographically by
``(cost, segments, predecessor, source)``.

The ``continuity`` objective first keeps the minimum total cost. Among those
plans it minimizes source changes (the first segment is not a change), then
segment count, final-segment start, and final source. At each recurrence, a
tie in these values uses the selected prefix's own continuity ordering. It is
computed with separate A/B prefix states; the no-source sentinel is only used
for a first segment and is never inserted into an A/B state window.

Two useful facts about the cap hold and are checked against an independent
enumeration oracle:

* A feasible cap never changes the same-cost adjudication. Under the default
  objective the segment count is itself the first secondary key, so the
  unbounded winner is a minimum-segment member of the optimal set. Under
  continuity the lexicographically smallest (switches, segments, ...) plan
  also uses the minimum number of segments among all minimum-cost plans:
  with the segment cap still feasible, the selected source sequence has a
  minimum-switch cover and splitting an existing segment only raises the
  segment count; it can never reduce switches.
* The cap can of course change the *cost* (the unbounded optimum may use more
  segments than the cap allows); everything is then recomputed from the
  capped optimal set, including certainty.

Every successful solution also carries a per-position ``certainty`` label:
``A_ONLY`` / ``B_ONLY`` / ``EITHER``. Source ``s`` is "possible" at position
``k`` exactly when some globally minimum-cost legal cover (subject to the
segment cap, when one is given) contains a source-``s`` segment covering
``k``. With a cap this set is objective-independent for the same reason as
the unbounded one -- the continuity switch count and the other secondary tie
keys only pick one plan out of the capped minimum-cost set and never narrow
it -- so both objectives report the same labels; the labels may still differ
from the uncapped ones, because the two optimal sets differ.

The unbounded certainty computation is a pure minimum-cost prefix/suffix
split. A source-``s`` segment ``[i, j)`` can occur in a minimum-cost cover
iff

    f[i] + fee[s] + (prefix_s[j] - prefix_s[i]) + g[j] == f[n]

where ``f[i]`` is the cheapest cost of covering ``[0, i)`` (over all sources)
and ``g[j]`` the cheapest cost of covering ``[j, n)`` (over all sources);
note the suffix does not depend on the source of the segment ending at ``j``.
For fixed ``(j, s)`` the minimizing prefixes ``i`` in the legal predecessor
range form a contiguous range; its leftmost member is a sliding-window
minimum of ``f[i] - prefix_s[i]`` and the range is merged into a
difference-array sweep. No segmentation is enumerated and winning-
predecessor replay is not enough, as it cannot reveal alternative optimal
predecessors.

With a cap the same split gains a used-segment dimension. ``f_c[i]`` is the
cheapest cover of ``[0, i)`` using exactly ``c`` segments (and ``F_c`` the
cheapest using at most ``c``); ``g_c[j]`` is the cheapest cover of
``[j, n)`` using at most ``c`` segments. Segment ``[i, j)`` of source ``s``
is possible in a capped optimum iff for some ``c`` in ``1..max_segments`` the
sliding minimum of ``f_c[i] - prefix_s[i]`` over the legal predecessor range
equals ``F_max[n] - g_{max-c}[j] - fee[s] - prefix_s[j]``. Every layer is a
deque sweep of the same shape as the unbounded one, so the capped work is
O(max_segments * (n + m)); with max_segments <= 8 the n=200_000 scale stays
linear in practice. Unbounded paths remain O(n + m) time and O(n) space
where ``m`` is the total number of unavailable windows; positions and windows
are never combined into a Cartesian product.
"""

from array import array
from collections import deque
from dataclasses import dataclass, field

# Source ids double as tie-breakers: A < B.
SOURCE_A = 0
SOURCE_B = 1
SOURCE_NAMES = ("A", "B")

OBJECTIVE_DEFAULT = "default"
OBJECTIVE_CONTINUITY = "continuity"

_NO_SOURCE = -1

# Costs are bounded by n * 1_000_000 + n * 1_000_000 < 10**12; this sentinel
# is safely beyond every finite plan cost.
_INF = 10**30

# Compact int64 storage for the capped recurrences (at most 9 layers of
# n+1 values each). 10**15 exceeds every plan cost (< 4*10**11) and fits the
# signed 64-bit range used by array("q").
_INF_Q = 10**15

# Tightest/freest segment-count limits accepted by the API layer.
MAX_SEGMENTS_MIN = 1
MAX_SEGMENTS_MAX = 8

CERTAINTY_A_ONLY = "A_ONLY"
CERTAINTY_B_ONLY = "B_ONLY"
CERTAINTY_EITHER = "EITHER"


class NoFeasibleRepair(Exception):
    """Raised when availability windows rule out every complete cover."""


class SegmentLimitExceeded(Exception):
    """Raised when a legal cover exists but none respects max_segments.

    A reachability pass ignoring cost establishes that some legal cover does
    exist; its minimum required segment count is strictly greater than the
    cap. Kept distinct from :class:`NoFeasibleRepair` so the API can report
    the concrete reason instead of a generic failure.
    """


@dataclass(frozen=True)
class Segment:
    """A half-open segment [start, end) served by one source."""

    start: int
    end: int
    source: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "source": self.source}


@dataclass(frozen=True)
class Solution:
    cost: int
    segments: tuple[Segment, ...]
    certainty: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Availability:
    """Per-source blocked-position lookup tables.

    ``last_down[s][j]`` is the largest blocked position ``p < j`` of source
    ``s`` (``-1`` when there is none); a source-``s`` segment ``[i, j)`` is
    legal iff ``i > last_down[s][j]`` and ``i >= j - L``.

    ``next_down[s][j]`` is the smallest blocked position ``p >= j`` (``n``
    when there is none); a source-``s`` segment ``[j, k)`` is legal iff
    ``k <= next_down[s][j]`` and ``k <= j + L``.
    """

    last_down: tuple[list[int], list[int]]
    next_down: tuple[list[int], list[int]]


def _build_availability(
    n: int, unavailable
) -> Availability:
    """Build the blocked-position tables from validated half-open windows.

    ``unavailable`` is ``(windows_a, windows_b)``; each list holds strictly
    increasing, non-overlapping, non-touching ``(start, end)`` pairs. The
    API layer validates them, so per-source coverage is a 0/1 difference
    sweep rather than a positions-times-windows product.
    """
    if unavailable is None:
        unavailable = ((), ())

    last_tables: list[list[int]] = []
    next_tables: list[list[int]] = []
    for windows in unavailable:
        mark = [0] * (n + 1)
        for start, end in windows:
            mark[start] += 1
            mark[end] -= 1

        blocked = bytearray(n)
        active = 0
        for k in range(n):
            active += mark[k]
            blocked[k] = 1 if active > 0 else 0

        last = [-1] * (n + 1)
        for j in range(1, n + 1):
            last[j] = j - 1 if blocked[j - 1] else last[j - 1]

        nxt = [n] * (n + 1)
        for j in range(n - 1, -1, -1):
            nxt[j] = j if blocked[j] else nxt[j + 1]

        last_tables.append(last)
        next_tables.append(nxt)

    return Availability(
        last_down=(last_tables[SOURCE_A], last_tables[SOURCE_B]),
        next_down=(next_tables[SOURCE_A], next_tables[SOURCE_B]),
    )


def solve(n: int, costs: list[tuple[int, int]], fee_a: int, fee_b: int,
          max_len: int, objective: str = OBJECTIVE_DEFAULT,
          unavailable=None, max_segments: int | None = None,
          prefix=None) -> Solution:
    """Compute the optimal cover of [0, n) plus per-position certainty.

    ``costs[k]`` is the pair of per-position costs ``(cost of A, cost of
    B)`` at position ``k``. ``unavailable`` is an optional
    ``(windows_a, windows_b)`` pair of half-open ``(start, end)`` windows.
    ``max_segments`` optionally bounds the number of segments; the API layer
    guarantees it is an integer in ``[1, 8]`` when given. ``prefix`` is an
    optional ordered tuple of executed ``(start, end, source_id)`` segments
    tiling ``[0, p)``; the API layer guarantees the tiling, length and
    availability checks already passed. Inputs are assumed otherwise
    validated by the API layer; the algorithm itself trusts the bounds.

    Raises :class:`NoFeasibleRepair` if availability alone rules out every
    legal cover, or :class:`SegmentLimitExceeded` if a legal cover exists but
    every one uses more than ``max_segments`` segments.
    """
    fees = (fee_a, fee_b)
    prefixes = _prefix_sums(n, costs)
    availability = _build_availability(n, unavailable)

    fixed: tuple[tuple[int, int, int], ...] = prefix or ()
    if not fixed:
        return _solve_full(
            n, prefixes, fees, max_len, availability, objective,
            max_segments)

    # The executed prefix is fixed: reprice it from the rates instead of
    # trusting any caller-supplied cost.
    prefix_cost = 0
    prefix_switches = 0
    previous_source = _NO_SOURCE
    for start, end, source_id in fixed:
        prefix_cost += fees[source_id] \
            + prefixes[source_id][end] - prefixes[source_id][start]
        if previous_source != _NO_SOURCE and source_id != previous_source:
            prefix_switches += 1
        previous_source = source_id
    p = fixed[-1][1]
    prefix_count = len(fixed)
    last_source = previous_source

    labels = [""] * n
    for start, end, source_id in fixed:
        name = CERTAINTY_A_ONLY if source_id == SOURCE_A \
            else CERTAINTY_B_ONLY
        for k in range(start, end):
            labels[k] = name

    if max_segments is None:
        if p < n and _min_segment_counts_suffix(
                n, max_len, availability, p) is None:
            raise NoFeasibleRepair()
        if p == n:
            return Solution(
                prefix_cost,
                tuple(Segment(start, end, SOURCE_NAMES[source_id])
                      for start, end, source_id in fixed),
                tuple(labels))
        if objective == OBJECTIVE_CONTINUITY:
            suffix = _solve_continuity_suffix(
                n, prefixes, fees, max_len, availability, p,
                prefix_cost, prefix_count, prefix_switches, last_source)
        else:
            suffix = _solve_default_suffix(
                n, prefixes, fees, max_len, availability, p,
                prefix_cost, prefix_count)
        diff = _certainty_suffix(
            n, prefixes, fees, max_len, availability, p,
            prefix_cost, suffix.cost)
        _merge_certainty_labels(labels, diff, n)
        segments = tuple(
            Segment(start, end, SOURCE_NAMES[source_id])
            for start, end, source_id in fixed
        ) + suffix.segments
        return Solution(suffix.cost, segments, tuple(labels))

    remaining_budget = max_segments - prefix_count
    # Zero new segments are needed when p == n; a negative remaining budget
    # means even the executed work alone exceeds the roster cap.
    min_needed = 0 if p == n else (
        _min_segment_counts_suffix(n, max_len, availability, p))
    if min_needed is None:
        raise NoFeasibleRepair()
    if min_needed > remaining_budget:
        raise SegmentLimitExceeded()

    if p == n:
        return Solution(
            prefix_cost,
            tuple(Segment(start, end, SOURCE_NAMES[source_id])
                  for start, end, source_id in fixed),
            tuple(labels))

    if objective == OBJECTIVE_CONTINUITY:
        suffix = _solve_capped_continuity_suffix(
            n, prefixes, fees, max_len, availability,
            max_segments, remaining_budget, p,
            prefix_cost, prefix_count, prefix_switches, last_source)
        exact_f = _capped_min_prefix_costs_suffix(
            n, prefixes, fees, max_len, availability,
            max_segments, remaining_budget, p, prefix_cost, prefix_count)
    else:
        suffix, exact_f = _solve_capped_default_suffix(
            n, prefixes, fees, max_len, availability,
            max_segments, remaining_budget, p,
            prefix_cost, prefix_count)

    diff = _certainty_capped_suffix(
        n, prefixes, fees, max_len, availability,
        max_segments, remaining_budget, exact_f, p,
        prefix_cost, prefix_count, suffix.cost)
    _merge_certainty_labels(labels, diff, n)
    segments = tuple(
        Segment(start, end, SOURCE_NAMES[source_id])
        for start, end, source_id in fixed
    ) + suffix.segments
    return Solution(suffix.cost, segments, tuple(labels))


def _solve_full(n, prefixes, fees, max_len, availability, objective,
                max_segments) -> Solution:
    """Legacy entry: no executed prefix, bit-identical pre-feature paths."""
    if max_segments is None:
        if objective == OBJECTIVE_CONTINUITY:
            solution = _solve_continuity(
                n, prefixes, fees, max_len, availability)
        else:
            solution = _solve_default(
                n, prefixes, fees, max_len, availability)
        certainty = _certainty(n, prefixes, fees, max_len, availability)
        return Solution(solution.cost, solution.segments, tuple(certainty))

    # The cap is part of feasibility itself: first the cost-blind segment
    # minimum decides between "no cover at all" and "covers need too many
    # segments", before any optimum is computed.
    min_needed = _min_segment_counts(n, max_len, availability)
    if min_needed is None:
        raise NoFeasibleRepair()
    if min_needed > max_segments:
        raise SegmentLimitExceeded()

    if objective == OBJECTIVE_CONTINUITY:
        solution = _solve_capped_continuity(
            n, prefixes, fees, max_len, availability, max_segments)
        # The adjudication states carry cost too, but certainty keeps its
        # own cost-only tables: it must not depend on tie-break state.
        exact_f = _capped_min_prefix_costs(
            n, prefixes, fees, max_len, availability, max_segments)
    else:
        solution, exact_f = _solve_capped_default(
            n, prefixes, fees, max_len, availability, max_segments)

    certainty = _certainty_capped(
        n, prefixes, fees, max_len, availability, max_segments, exact_f)
    return Solution(solution.cost, solution.segments, tuple(certainty))


def _merge_certainty_labels(labels: list[str], diff, n: int) -> None:
    """Merge per-source difference-array coverage into the label array.

    Positions already labelled (the executed prefix) are fixed to their
    source and are never touched; the difference sweep only contributes on
    the remaining suffix.
    """
    active = [0, 0]
    for k in range(n):
        active[SOURCE_A] += diff[SOURCE_A][k]
        active[SOURCE_B] += diff[SOURCE_B][k]
        if labels[k] != "":
            continue
        a_possible = active[SOURCE_A] > 0
        b_possible = active[SOURCE_B] > 0
        if a_possible and b_possible:
            labels[k] = CERTAINTY_EITHER
        elif a_possible:
            labels[k] = CERTAINTY_A_ONLY
        elif b_possible:
            labels[k] = CERTAINTY_B_ONLY
        else:  # pragma: no cover - every position is covered by an optimum
            raise AssertionError("position uncovered by all optimal plans")


def _prefix_sums(
    n: int, costs: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    prefix_a = [0] * (n + 1)
    prefix_b = [0] * (n + 1)
    for k in range(n):
        a, b = costs[k]
        prefix_a[k + 1] = prefix_a[k] + a
        prefix_b[k + 1] = prefix_b[k] + b
    return prefix_a, prefix_b


def _solve_default(n: int, prefixes: tuple[list[int], list[int]],
                   fees: tuple[int, int], max_len: int,
                   availability: Availability) -> Solution:
    # DP tables. Unreachable endpoints keep _INF and are never enqueued.
    best_cost = [_INF] * (n + 1)
    best_cost[0] = 0
    best_seg_count = [0] * (n + 1)
    prev_index = [-1] * (n + 1)
    prev_source = [-1] * (n + 1)

    # Each entry in window s is an index i, keyed by
    # (best_cost[i] - prefix_s[i], best_seg_count[i], i). Only indices from
    # which an uninterrupted source-s run reaches j can survive at the front.
    windows: tuple[deque, deque] = (deque([0]), deque([0]))

    for j in range(1, n + 1):
        best_candidate = None  # (cost, segments, prev, source)
        for s, window in enumerate(windows):
            # Segment [i, j) must fit the length cap and strictly pass the
            # last blocked position of s before j.
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if not window:
                continue
            i = window[0]
            candidate = (
                best_cost[i] - prefixes[s][i] + prefixes[s][j] + fees[s],
                best_seg_count[i] + 1,
                i,
                s,
            )
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        if best_candidate is None:
            # j cannot be covered; it must not become anyone's predecessor.
            continue

        cost, seg_count, i, s = best_candidate
        best_cost[j] = cost
        best_seg_count[j] = seg_count
        prev_index[j] = i
        prev_source[j] = s

        for s, window in enumerate(windows):
            key = (best_cost[j] - prefixes[s][j], best_seg_count[j])
            while window:
                tail = window[-1]
                tail_key = (
                    best_cost[tail] - prefixes[s][tail],
                    best_seg_count[tail],
                )
                # Equal keys retain the smaller/older index; it has the
                # predecessor-index tie breaker and expires first.
                if tail_key <= key:
                    break
                window.pop()
            window.append(j)

    if best_cost[n] >= _INF:
        raise NoFeasibleRepair()
    return _reconstruct(n, prev_index, prev_source, best_cost[n])


def _solve_continuity(n: int, prefixes: tuple[list[int], list[int]],
                      fees: tuple[int, int], max_len: int,
                      availability: Availability) -> Solution:
    # State p is the source of the last segment (0=A, 1=B). Arrays are
    # indexed [p][j]. The empty prefix is not state A or B.
    best_cost = [[_INF] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    switches = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    seg_count = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    last_start = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    full_keys = [[()] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_index = [[-1] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_state = [[_NO_SOURCE] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]

    # windows[p][s] contains state-p prefixes to which a source-s segment is
    # appended. The two no-source rows hold only index 0 until it expires.
    windows = tuple(
        tuple(deque([0]) for _s in (SOURCE_A, SOURCE_B))
        for _p in range(3)
    )

    for j in range(1, n + 1):
        for s in (SOURCE_A, SOURCE_B):
            # Availability of the appended source s bounds every row's
            # window identically: [max(j-L, last_down_s[j]+1), j-1].
            low = max(j - max_len, availability.last_down[s][j] + 1)

            best_candidate = None
            chosen_predecessor = _NO_SOURCE
            for p in range(3):
                window = windows[p][s]
                while window and window[0] < low:
                    window.popleft()
                if not window:
                    continue
                i = window[0]
                if p < 2:
                    if best_cost[p][i] >= _INF:
                        continue
                    prefix_cost = best_cost[p][i]
                    prefix_switches = switches[p][i]
                    prefix_count = seg_count[p][i]
                    # The new final source is fixed, so a tie in the visible
                    # final-plan keys is resolved with the prefix's own full
                    # continuity order.
                    prefix_tie = full_keys[p][i]
                    predecessor_state = p
                    added_switch = 0 if p == s else 1
                else:
                    prefix_cost = 0
                    prefix_switches = 0
                    prefix_count = 0
                    prefix_tie = ()
                    predecessor_state = _NO_SOURCE
                    added_switch = 0

                candidate = (
                    prefix_cost - prefixes[s][i] + prefixes[s][j] + fees[s],
                    prefix_switches + added_switch,
                    prefix_count + 1,
                    i,
                    s,
                    prefix_tie,
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
                    chosen_predecessor = predecessor_state

            if best_candidate is None:
                # No legal source-s segment ends at j; state (s, j) stays
                # unreachable and is inserted into no predecessor window.
                continue

            cost, switch_count, count, i, s, _prefix_tie = best_candidate
            best_cost[s][j] = cost
            switches[s][j] = switch_count
            seg_count[s][j] = count
            last_start[s][j] = i
            full_keys[s][j] = best_candidate
            prev_index[s][j] = i
            prev_state[s][j] = chosen_predecessor

            # A real A/B prefix becomes a predecessor only in windows keyed
            # by its actual last-source state.
            for new_s, window in enumerate(windows[s]):
                key = (
                    best_cost[s][j] - prefixes[new_s][j],
                    switches[s][j],
                    seg_count[s][j],
                )
                while window:
                    tail = window[-1]
                    tail_key = (
                        best_cost[s][tail] - prefixes[new_s][tail],
                        switches[s][tail],
                        seg_count[s][tail],
                    )
                    if tail_key <= key:
                        break
                    window.pop()
                window.append(j)

    final_candidate = None
    final_state = SOURCE_A
    for p in (SOURCE_A, SOURCE_B):
        if best_cost[p][n] >= _INF:
            continue
        candidate = (
            best_cost[p][n],
            switches[p][n],
            seg_count[p][n],
            last_start[p][n],
            p,
        )
        if final_candidate is None or candidate < final_candidate:
            final_candidate = candidate
            final_state = p

    if final_candidate is None:
        raise NoFeasibleRepair()
    return _reconstruct_states(
        n, prev_index, prev_state, final_state, final_candidate[0])


def _reconstruct(n: int, prev_index: list[int], prev_source: list[int],
                 total_cost: int) -> Solution:
    segments: list[Segment] = []
    end = n
    while end > 0:
        start = prev_index[end]
        segments.append(Segment(start, end, SOURCE_NAMES[prev_source[end]]))
        end = start
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


def _reconstruct_states(n: int, prev_index: list[list[int]],
                        prev_state: list[list[int]], final_state: int,
                        total_cost: int) -> Solution:
    segments: list[Segment] = []
    end = n
    state = final_state
    while end > 0:
        start = prev_index[state][end]
        segments.append(Segment(start, end, SOURCE_NAMES[state]))
        next_state = prev_state[state][end]
        end = start
        state = next_state
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


# --------------------------------------------------------------------------
# Per-position certainty over the set of globally minimum-cost covers
# --------------------------------------------------------------------------


def _min_prefix_costs(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability
) -> list[int]:
    """f[j] = minimum cost of any legal cover of [0, j).

    Cost-only version of the default recurrence. Each per-source window is a
    monotone deque of indices ordered by f[i] - prefix_s[i]; expired indices
    leave at the front. Unreachable endpoints stay at _INF.
    """
    f = [_INF] * (n + 1)
    f[0] = 0
    windows = (deque([0]), deque([0]))
    for j in range(1, n + 1):
        best = None
        for s, window in enumerate(windows):
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if not window:
                continue
            i = window[0]
            value = f[i] - prefixes[s][i] + prefixes[s][j] + fees[s]
            if best is None or value < best:
                best = value
        if best is None:
            continue
        f[j] = best
        for s, window in enumerate(windows):
            value = f[j] - prefixes[s][j]
            while window:
                tail = window[-1]
                if f[tail] - prefixes[s][tail] <= value:
                    break
                window.pop()
            window.append(j)
    return f


def _min_suffix_costs(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability
) -> list[int]:
    """g[j] = minimum cost of any legal cover of [j, n).

    The recurrence g[j] = min over (s, k) with j < k <= min(n, j+L,
    next_down_s[j]) of fee[s] + prefix_s[k] - prefix_s[j] + g[k] rewrites,
    for a fixed successor k, as g[k] + fee[s] + prefix_s[k] minus
    prefix_s[j]. Indices k enter the window in descending order, so the
    monotone deques store indices in descending order and expiry
    (k past the length cap or the next blocked position) happens at front.
    """
    g = [_INF] * (n + 1)
    g[n] = 0
    windows = (deque([n]), deque([n]))
    for j in range(n - 1, -1, -1):
        best = None
        for s, window in enumerate(windows):
            high = min(j + max_len, availability.next_down[s][j])
            while window and window[0] > high:
                window.popleft()
            if not window:
                continue
            k = window[0]
            value = g[k] + fees[s] + prefixes[s][k] - prefixes[s][j]
            if best is None or value < best:
                best = value
        if best is None:
            continue
        g[j] = best
        for s, window in enumerate(windows):
            value = g[j] + fees[s] + prefixes[s][j]
            while window:
                tail = window[-1]
                if g[tail] + fees[s] + prefixes[s][tail] <= value:
                    break
                window.pop()
            window.append(j)
    return g


def _certainty(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability
) -> list[str]:
    """Label every position by which sources occur in minimum-cost covers.

    A source-s segment [i, j) participates in an optimal cover exactly when

        f[i] + fee[s] + P_s[j] - P_s[i] + g[j] == f[n]

    i.e. h_s(i) = f[i] - P_s[i] equals target = f[n] - g[j] - fee[s] -
    P_s[j]. Concatenating any prefix, the segment and any suffix is itself a
    complete feasible cover, so h_s(i) >= target for every feasible i (its
    cover costs at least the optimum); the segment is possible iff the
    window minimum equals target. Among ties the deque keeps the smallest
    predecessor, whose single segment [i_min, j) already covers the union of
    every minimizing predecessor's interval. Availability narrows both the
    prefix window (last blocked position) and the suffix reachability test;
    it is never applied after the fact.
    """
    f = _min_prefix_costs(n, prefixes, fees, max_len, availability)
    g = _min_suffix_costs(n, prefixes, fees, max_len, availability)
    if f[n] >= _INF:
        raise NoFeasibleRepair()
    optimum = f[n]

    # Per source: a monotone deque of predecessor indices ordered by
    # h_s(i) = f[i] - P_s[i]. Equal keys are retained (oldest first) so the
    # front is the smallest index attaining the window minimum.
    windows = (deque([0]), deque([0]))

    # difference[s] marks coverage of positions by source s over [0, n).
    difference = ([0] * (n + 1), [0] * (n + 1))

    for j in range(1, n + 1):
        for s, window in enumerate(windows):
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if not window or g[j] >= _INF:
                continue
            i = window[0]
            target = optimum - g[j] - fees[s] - prefixes[s][j]
            if f[i] - prefixes[s][i] == target:
                diff = difference[s]
                diff[i] += 1
                diff[j] -= 1

        if f[j] < _INF:
            for s, window in enumerate(windows):
                key = f[j] - prefixes[s][j]
                while window and \
                        f[window[-1]] - prefixes[s][window[-1]] > key:
                    window.pop()
                window.append(j)

    labels = [""] * n
    active = [0, 0]
    for k in range(n):
        active[SOURCE_A] += difference[SOURCE_A][k]
        active[SOURCE_B] += difference[SOURCE_B][k]
        a_possible = active[SOURCE_A] > 0
        b_possible = active[SOURCE_B] > 0
        if a_possible and b_possible:
            labels[k] = CERTAINTY_EITHER
        elif a_possible:
            labels[k] = CERTAINTY_A_ONLY
        elif b_possible:
            labels[k] = CERTAINTY_B_ONLY
        else:  # pragma: no cover - every position is covered by an optimum
            raise AssertionError("position uncovered by all optimal plans")
    return labels



# --------------------------------------------------------------------------
# Segment-count cap: feasibility, cost tables, adjudication, certainty
#
# Every capped recurrence layers its state by the exact segment count c.
# Within one layer a sweep only ever reads endpoints of the *previous* layer
# (layer 1 only reads the empty prefix at index 0); predecessor endpoints are
# activated lazily, j-1 right before endpoint j is evaluated, so a layer can
# never read its own endpoints and no (c, j) state is reachable with a count
# different from the layer label.
# --------------------------------------------------------------------------


def _q_table(n: int, fill: int) -> array:
    table = array("q", [fill])
    table.extend([fill] * n)
    return table


def _min_segment_counts(
    n: int, max_len: int, availability: Availability
):
    """Minimum legal segment count reaching ``n``, ignoring cost.

    A standard sliding-window reachability DP: one FIFO deque per source
    holding reachable indices ordered by increasing index (and therefore
    non-decreasing minimum count). Unlike the cost DPs, equal keys dominate
    nothing here -- a higher-count predecessor is the only reachable one once
    a longer stretch becomes illegal -- so every reachable index is appended
    and only expired at the front. The minimum is computed without pruning at
    the cap: an endpoint needing more than ``limit`` segments may still be an
    indispensable intermediate (with ``L=1`` endpoint ``j`` is always reached
    in exactly ``j`` segments). The returned value is compared to ``limit``
    by the caller, which is exactly what separates "no legal cover exists"
    from "every cover needs more segments than allowed". Returns ``None`` when
    ``n`` is unreachable.
    """
    need = [-1] * (n + 1)
    need[0] = 0
    windows = (deque([0]), deque([0]))
    for j in range(1, n + 1):
        best = None
        for s, window in enumerate(windows):
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if window:
                value = need[window[0]] + 1
                if best is None or value < best:
                    best = value
        if best is not None:
            need[j] = best
            for window in windows:
                window.append(j)
    return None if need[n] < 0 else need[n]


def _activate_capped_prefix(window: deque, index: int,
                            previous_cost: array, prefix_s: list[int]):
    """Insert one previous-layer endpoint into a source prefix deque."""
    value = previous_cost[index]
    if value >= _INF_Q:
        return
    key = value - prefix_s[index]
    while window and \
            previous_cost[window[-1]] - prefix_s[window[-1]] > key:
        window.pop()
    window.append(index)


def _capped_min_prefix_costs(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int,
) -> list[array]:
    """``f[c][j]`` = cheapest cover of ``[0, j)`` with exactly ``c`` segments.

    Only ``f[0][0]`` is finite. Layer 1 activates the empty prefix alone;
    later layers activate endpoints ``1..n-1`` of the previous layer, never
    index 0 (appending to the empty prefix would under-count segments).
    """
    f = [_q_table(n, _INF_Q) for _ in range(max_segments + 1)]
    f[0][0] = 0

    for c in range(1, max_segments + 1):
        previous = f[c - 1]
        current = f[c]
        windows = (deque(), deque())
        for j in range(1, n + 1):
            predecessor = j - 1
            if c == 1:
                if predecessor == 0:
                    for s, window in enumerate(windows):
                        _activate_capped_prefix(
                            window, 0, previous, prefixes[s])
            elif predecessor >= 1:
                for s, window in enumerate(windows):
                    _activate_capped_prefix(
                        window, predecessor, previous, prefixes[s])

            best = _INF_Q
            for s, window in enumerate(windows):
                low = max(j - max_len, availability.last_down[s][j] + 1)
                while window and window[0] < low:
                    window.popleft()
                if window:
                    i = window[0]
                    value = previous[i] - prefixes[s][i] \
                        + prefixes[s][j] + fees[s]
                    if value < best:
                        best = value
            if best < _INF_Q:
                current[j] = best
    return f


def _capped_min_suffix_costs(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int,
) -> list[array]:
    """``g[c][j]`` = cheapest cover of ``[j, n)`` using at most ``c``
    segments.

    Exact-count rows are swept in decreasing index order (mirroring the
    unbounded suffix DP), then accumulated into at-most-c budgets. Only
    ``g[c][n]`` is 0 (the empty suffix fits every budget).
    """
    exact = [_q_table(n, _INF_Q) for _ in range(max_segments + 1)]
    exact[0][n] = 0

    for c in range(1, max_segments + 1):
        previous = exact[c - 1]
        current = exact[c]
        if c == 1:
            # Endpoint n seeds the one-segment windows from the outset, just
            # as index 0 seeds the first prefix layer: [j, n) itself may be a
            # single segment for every j within L of n.
            windows = (deque([n]), deque([n]))
        else:
            # Transitions ending exactly at n need exact[c-1] cover of the
            # empty suffix, which is infinite for c-1 >= 1, so n never
            # enters later layers.
            windows = (deque(), deque())
        for j in range(n - 1, -1, -1):
            successor = j + 1
            if c >= 2 and successor <= n - 1:
                for s, window in enumerate(windows):
                    if previous[successor] < _INF_Q:
                        key = previous[successor] + fees[s] \
                            + prefixes[s][successor]
                        while window and previous[window[-1]] + fees[s] \
                                + prefixes[s][window[-1]] > key:
                            window.pop()
                        window.append(successor)

            best = _INF_Q
            for s, window in enumerate(windows):
                high = min(j + max_len, availability.next_down[s][j])
                while window and window[0] > high:
                    window.popleft()
                if window:
                    k = window[0]
                    value = previous[k] + fees[s] \
                        + prefixes[s][k] - prefixes[s][j]
                    if value < best:
                        best = value
            if best < _INF_Q:
                current[j] = best

    budget = [_q_table(n, _INF_Q) for _ in range(max_segments + 1)]
    for j in range(n + 1):
        running = _INF_Q
        for c in range(max_segments + 1):
            if exact[c][j] < running:
                running = exact[c][j]
            budget[c][j] = running
    return budget


def _certainty_capped(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int, exact_f: list[array],
) -> list[str]:
    """Per-position source union over cap-respecting minimum-cost plans.

    A source-``s`` segment ``[i, j)`` appended to an exactly-``c``-segment
    prefix occurs in a capped optimum iff

        f[c][i] - P_s[i] + P_s[j] + fee[s] + g[M-c][j] == F[M][n]

    where ``f`` counts prefix segments exactly, ``F`` by budget and ``g`` by
    remaining budget. Each ``(c, s)`` window keeps the smallest predecessor
    attaining the window minimum; its single interval ``[i_min, j)`` already
    covers the union of every minimizing predecessor's interval. Concatenating
    the cheapest exact-c prefix, the segment and any cheapest suffix is itself
    a complete capped-optimum plan, so membership is exact.
    """
    budget_f = [_q_table(n, _INF_Q) for _ in range(max_segments + 1)]
    for j in range(n + 1):
        running = _INF_Q
        for c in range(max_segments + 1):
            if exact_f[c][j] < running:
                running = exact_f[c][j]
            budget_f[c][j] = running

    g = _capped_min_suffix_costs(
        n, prefixes, fees, max_len, availability, max_segments)

    optimum = budget_f[max_segments][n]
    if optimum >= _INF_Q:  # pragma: no cover - feasibility prechecked
        raise NoFeasibleRepair()

    # windows[c][s]: exact-c prefix predecessors. Layer 0 stays empty; the
    # empty prefix is activated into layer 1 during the sweep below.
    windows = tuple(
        (deque(), deque()) for _ in range(max_segments + 1)
    )
    difference = ([0] * (n + 1), [0] * (n + 1))

    for j in range(1, n + 1):
        predecessor = j - 1
        for c in range(1, max_segments + 1):
            if c == 1:
                if predecessor == 0 and exact_f[0][0] < _INF_Q:
                    for s, window in enumerate(windows[c]):
                        _activate_capped_prefix(
                            window, 0, exact_f[0], prefixes[s])
            elif predecessor >= 1 and \
                    exact_f[c - 1][predecessor] < _INF_Q:
                for s, window in enumerate(windows[c]):
                    _activate_capped_prefix(
                        window, predecessor, exact_f[c - 1], prefixes[s])

            remaining = g[max_segments - c]
            if remaining[j] >= _INF_Q:
                continue
            for s, window in enumerate(windows[c]):
                low = max(j - max_len, availability.last_down[s][j] + 1)
                while window and window[0] < low:
                    window.popleft()
                if not window:
                    continue
                i = window[0]
                target = optimum - remaining[j] - fees[s] - prefixes[s][j]
                # The window stores endpoints of the previous (exact-c-1)
                # layer, so the optimality test reads that row too.
                if exact_f[c - 1][i] - prefixes[s][i] == target:
                    diff = difference[s]
                    diff[i] += 1
                    diff[j] -= 1

    labels = [""] * n
    active = [0, 0]
    for k in range(n):
        active[SOURCE_A] += difference[SOURCE_A][k]
        active[SOURCE_B] += difference[SOURCE_B][k]
        a_possible = active[SOURCE_A] > 0
        b_possible = active[SOURCE_B] > 0
        if a_possible and b_possible:
            labels[k] = CERTAINTY_EITHER
        elif a_possible:
            labels[k] = CERTAINTY_A_ONLY
        elif b_possible:
            labels[k] = CERTAINTY_B_ONLY
        else:  # pragma: no cover - every position is covered by an optimum
            raise AssertionError("position uncovered by all capped optima")
    return labels


def _solve_capped_default(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int,
):
    """Default objective under an at-most-``max_segments`` cap.

    The segment layer supplies the secondary segment-count key, so inside one
    layer candidates compare by ``(cost, predecessor, source)``. The final
    plan minimizes ``(cost, layer)``. Per-layer backpointers are compact
    32-bit/8-bit arrays; the exact-count cost tables are reused by certainty.
    """
    exact_f = [_q_table(n, _INF_Q) for _ in range(max_segments + 1)]
    exact_f[0][0] = 0
    prev_index = [
        array("i", [-1] * (n + 1)) for _ in range(max_segments + 1)
    ]
    prev_source = [
        array("b", [_NO_SOURCE] * (n + 1))
        for _ in range(max_segments + 1)
    ]

    for c in range(1, max_segments + 1):
        previous = exact_f[c - 1]
        windows = (deque(), deque())
        for j in range(1, n + 1):
            predecessor = j - 1
            if c == 1:
                if predecessor == 0:
                    for s, window in enumerate(windows):
                        _activate_capped_prefix(
                            window, 0, previous, prefixes[s])
            elif predecessor >= 1 and previous[predecessor] < _INF_Q:
                for s, window in enumerate(windows):
                    _activate_capped_prefix(
                        window, predecessor, previous, prefixes[s])

            best_candidate = None  # (cost, predecessor, source)
            for s, window in enumerate(windows):
                low = max(j - max_len, availability.last_down[s][j] + 1)
                while window and window[0] < low:
                    window.popleft()
                if not window:
                    continue
                i = window[0]
                candidate = (
                    previous[i] - prefixes[s][i]
                    + prefixes[s][j] + fees[s],
                    i,
                    s,
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
            if best_candidate is None:
                continue
            cost, i, s = best_candidate
            exact_f[c][j] = cost
            prev_index[c][j] = i
            prev_source[c][j] = s

    final_layer = -1
    final_cost = _INF_Q
    for c in range(1, max_segments + 1):
        if exact_f[c][n] < final_cost:
            final_cost = exact_f[c][n]
            final_layer = c
    if final_layer < 0:  # pragma: no cover - feasibility prechecked
        raise SegmentLimitExceeded()

    segments = []
    end = n
    layer = final_layer
    while end > 0:
        s = prev_source[layer][end]
        start = prev_index[layer][end]
        segments.append(Segment(start, end, SOURCE_NAMES[s]))
        end = start
        layer -= 1
    segments.reverse()
    return Solution(cost=final_cost, segments=tuple(segments)), exact_f


def _solve_capped_continuity(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int,
) -> Solution:
    """Continuity objective layered by exact segment count.

    It is the unbounded continuity recurrence with an outer layer sweep:
    predecessor states come exclusively from the previous layer, layer 1 is
    fed by the no-source sentinel only, and full recursive continuity keys
    decide A-vs-B predecessor ties at the same endpoint. Final selection
    compares ``(cost, switches, layer, last-start, final source)``.
    """
    final_candidate = None
    final_state = SOURCE_A
    final_layer = 1

    prev_index = [None] * (max_segments + 1)
    prev_state = [None] * (max_segments + 1)

    # Previous-layer per-state columns; absent for layer 1's sentinel.
    previous_cost = None
    previous_switches = None
    previous_keys = None

    for c in range(1, max_segments + 1):
        best_cost = [[_INF_Q] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
        switches = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
        full_keys = [[None] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
        layer_prev_index = [
            array("i", [-1] * (n + 1)) for _ in (SOURCE_A, SOURCE_B)
        ]
        layer_prev_state = [
            array("b", [_NO_SOURCE] * (n + 1))
            for _ in (SOURCE_A, SOURCE_B)
        ]

        if c == 1:
            # First segments [0, j) out of the empty prefix: one deque per
            # source, expiring exactly like an ordinary predecessor window.
            for s in (SOURCE_A, SOURCE_B):
                window = deque([0])
                for j in range(1, n + 1):
                    low = max(j - max_len,
                              availability.last_down[s][j] + 1)
                    while window and window[0] < low:
                        window.popleft()
                    if not window:
                        continue
                    candidate = (
                        prefixes[s][j] + fees[s], 0, 1, 0, s, (),
                    )
                    best_cost[s][j] = candidate[0]
                    switches[s][j] = 0
                    full_keys[s][j] = candidate
                    layer_prev_index[s][j] = 0
                    layer_prev_state[s][j] = _NO_SOURCE
        else:
            # windows[p][s]: append source s to a previous-layer state p.
            windows = tuple(
                tuple(deque() for _s in (SOURCE_A, SOURCE_B))
                for _p in (SOURCE_A, SOURCE_B)
            )
            for j in range(1, n + 1):
                predecessor = j - 1
                if predecessor >= 1:
                    for p in (SOURCE_A, SOURCE_B):
                        if previous_cost[p][predecessor] >= _INF_Q:
                            continue
                        for new_s, window in enumerate(windows[p]):
                            tail_key = (
                                previous_cost[p][predecessor]
                                - prefixes[new_s][predecessor],
                                previous_switches[p][predecessor],
                            )
                            while window:
                                tail = window[-1]
                                if (
                                    previous_cost[p][tail]
                                    - prefixes[new_s][tail],
                                    previous_switches[p][tail],
                                ) <= tail_key:
                                    break
                                window.pop()
                            window.append(predecessor)

                for s in (SOURCE_A, SOURCE_B):
                    low = max(j - max_len,
                              availability.last_down[s][j] + 1)
                    best_candidate = None
                    chosen_predecessor = _NO_SOURCE
                    for p in (SOURCE_A, SOURCE_B):
                        window = windows[p][s]
                        while window and window[0] < low:
                            window.popleft()
                        if not window:
                            continue
                        i = window[0]
                        candidate = (
                            previous_cost[p][i] - prefixes[s][i]
                            + prefixes[s][j] + fees[s],
                            previous_switches[p][i]
                            + (0 if p == s else 1),
                            c,
                            i,
                            s,
                            previous_keys[p][i],
                        )
                        if best_candidate is None \
                                or candidate < best_candidate:
                            best_candidate = candidate
                            chosen_predecessor = p
                    if best_candidate is None:
                        continue
                    cost, switch_count, _count, i, _s, _tie = \
                        best_candidate
                    best_cost[s][j] = cost
                    switches[s][j] = switch_count
                    full_keys[s][j] = best_candidate
                    layer_prev_index[s][j] = i
                    layer_prev_state[s][j] = chosen_predecessor

        prev_index[c] = layer_prev_index
        prev_state[c] = layer_prev_state
        previous_cost = best_cost
        previous_switches = switches
        previous_keys = full_keys

        for p in (SOURCE_A, SOURCE_B):
            if best_cost[p][n] >= _INF_Q:
                continue
            candidate = (
                best_cost[p][n], switches[p][n], c,
                layer_prev_index[p][n], p,
            )
            if final_candidate is None or candidate < final_candidate:
                final_candidate = candidate
                final_state = p
                final_layer = c

    if final_candidate is None:  # pragma: no cover - feasibility prechecked
        raise SegmentLimitExceeded()

    total_cost = final_candidate[0]
    segments = []
    end = n
    state = final_state
    layer = final_layer
    while end > 0:
        start = prev_index[layer][state][end]
        segments.append(Segment(start, end, SOURCE_NAMES[state]))
        next_state = prev_state[layer][state][end]
        end = start
        layer -= 1
        state = next_state
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


# --------------------------------------------------------------------------
# Executed-prefix suffix recurrences
#
# Every DP below is a suffix-seeded twin of the full-gap one: index p (not 0)
# is the single reachable seed, carries the fixed prefix's cost/segment
# count/last source, and only endpoints p+1..n are swept. The availability
# tables are unchanged. With p == 0 and an empty prefix these recurrences are
# exactly the legacy ones; the legacy functions remain the call path for that
# case, so omission of the request field cannot change a single value.
# --------------------------------------------------------------------------


def _min_segment_counts_suffix(
    n: int, max_len: int, availability: Availability, p: int
):
    """Minimum legal segment count covering ``[p, n)`` from seed ``p``.

    Suffix twin of :func:`_min_segment_counts`: FIFO deques per source with
    one reachable seed at index ``p`` (zero suffix segments). Returns
    ``None`` when ``n`` is unreachable. Costs are ignored, so this decides
    NO_FEASIBLE_REPAIR apart from the segment-limit failure.
    """
    need = [-1] * (n + 1)
    need[p] = 0
    windows = (deque([p]), deque([p]))
    for j in range(p + 1, n + 1):
        best = None
        for s, window in enumerate(windows):
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if window:
                value = need[window[0]] + 1
                if best is None or value < best:
                    best = value
        if best is not None:
            need[j] = best
            for window in windows:
                window.append(j)
    return None if need[n] < 0 else need[n]


def _solve_default_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    p: int, seed_cost: int, seed_count: int,
) -> Solution:
    """Default objective on ``[p, n)`` seeded with the executed prefix."""
    best_cost = [_INF] * (n + 1)
    best_cost[p] = seed_cost
    best_seg_count = [0] * (n + 1)
    best_seg_count[p] = seed_count
    prev_index = [-1] * (n + 1)
    prev_source = [-1] * (n + 1)

    windows = (deque([p]), deque([p]))
    for j in range(p + 1, n + 1):
        best_candidate = None
        for s, window in enumerate(windows):
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if not window:
                continue
            i = window[0]
            candidate = (
                best_cost[i] - prefixes[s][i] + prefixes[s][j] + fees[s],
                best_seg_count[i] + 1,
                i,
                s,
            )
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        if best_candidate is None:
            continue

        cost, seg_count, i, s = best_candidate
        best_cost[j] = cost
        best_seg_count[j] = seg_count
        prev_index[j] = i
        prev_source[j] = s

        for s, window in enumerate(windows):
            key = (best_cost[j] - prefixes[s][j], best_seg_count[j])
            while window:
                tail = window[-1]
                tail_key = (
                    best_cost[tail] - prefixes[s][tail],
                    best_seg_count[tail],
                )
                if tail_key <= key:
                    break
                window.pop()
            window.append(j)

    return _reconstruct_suffix(
        n, p, prev_index, prev_source, best_cost[n])


def _reconstruct_suffix(n: int, p: int, prev_index: list[int],
                        prev_source: list[int], total_cost: int) -> Solution:
    """Backtrack new-segment links only until the fixed boundary ``p``."""
    segments: list[Segment] = []
    end = n
    while end > p:
        start = prev_index[end]
        segments.append(Segment(start, end, SOURCE_NAMES[prev_source[end]]))
        end = start
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


def _solve_continuity_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    p: int, seed_cost: int, seed_count: int, seed_switches: int,
    seed_source: int,
) -> Solution:
    """Continuity objective on ``[p, n)`` seeded with the executed prefix.

    Unlike the empty-prefix recurrence the seed already has a last source
    (``seed_source``), so the first appended segment *does* count a switch
    when its source differs. There is no no-source sentinel row: the seed
    enters the ordinary per-state windows once, and its recursive tie key is
    the empty tuple (ties inside the fixed prefix cannot vary between
    completions and therefore never decide one).
    """
    best_cost = [[_INF] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    switches = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    seg_count = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    last_start = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    full_keys = [[()] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_index = [[-1] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
    prev_state = [[_NO_SOURCE] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]

    # windows[state][new_source]: seed states of final source ``state`` to
    # which a new_source segment is appended. Only the executed prefix's own
    # last source is seeded.
    windows = tuple(
        tuple(deque() for _s in (SOURCE_A, SOURCE_B))
        for _p in (SOURCE_A, SOURCE_B)
    )
    for new_s in (SOURCE_A, SOURCE_B):
        windows[seed_source][new_s].append(p)
    seed_columns = {
        seed_source: (
            seed_cost, seed_switches, seed_count, (),
        )
    }

    def column(state, index):
        if index == p:
            return seed_columns[state]
        return (best_cost[state][index], switches[state][index],
                seg_count[state][index], full_keys[state][index])

    for j in range(p + 1, n + 1):
        for s in (SOURCE_A, SOURCE_B):
            low = max(j - max_len, availability.last_down[s][j] + 1)

            best_candidate = None
            chosen_predecessor = _NO_SOURCE
            for prev_s in (SOURCE_A, SOURCE_B):
                window = windows[prev_s][s]
                while window and window[0] < low:
                    window.popleft()
                if not window:
                    continue
                i = window[0]
                prefix_cost, prefix_switches, prefix_count, prefix_tie = \
                    column(prev_s, i)
                if prefix_cost >= _INF:
                    continue
                candidate = (
                    prefix_cost - prefixes[s][i] + prefixes[s][j]
                    + fees[s],
                    prefix_switches + (0 if prev_s == s else 1),
                    prefix_count + 1,
                    i,
                    s,
                    prefix_tie,
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
                    chosen_predecessor = prev_s

            if best_candidate is None:
                continue

            cost, switch_count, count, i, _s, _prefix_tie = best_candidate
            best_cost[s][j] = cost
            switches[s][j] = switch_count
            seg_count[s][j] = count
            last_start[s][j] = i
            full_keys[s][j] = best_candidate
            prev_index[s][j] = i
            prev_state[s][j] = chosen_predecessor

            for new_s, window in enumerate(windows[s]):
                key = (
                    best_cost[s][j] - prefixes[new_s][j],
                    switches[s][j],
                    seg_count[s][j],
                )
                while window:
                    tail = window[-1]
                    tail_cost, tail_switches, tail_count, _ = \
                        column(s, tail)
                    tail_key = (
                        tail_cost - prefixes[new_s][tail],
                        tail_switches,
                        tail_count,
                    )
                    if tail_key <= key:
                        break
                    window.pop()
                window.append(j)

    final_candidate = None
    final_state = SOURCE_A
    for state in (SOURCE_A, SOURCE_B):
        if best_cost[state][n] >= _INF:
            continue
        candidate = (
            best_cost[state][n],
            switches[state][n],
            seg_count[state][n],
            last_start[state][n],
            state,
        )
        if final_candidate is None or candidate < final_candidate:
            final_candidate = candidate
            final_state = state

    if final_candidate is None:  # pragma: no cover - feasibility prechecked
        raise NoFeasibleRepair()
    return _reconstruct_states_suffix(
        n, p, prev_index, prev_state, final_state, final_candidate[0])


def _reconstruct_states_suffix(n: int, p: int, prev_index, prev_state,
                               final_state: int, total_cost: int) -> Solution:
    """Backtrack continuity state links down to boundary ``p``."""
    segments: list[Segment] = []
    end = n
    state = final_state
    while end > p:
        start = prev_index[state][end]
        segments.append(Segment(start, end, SOURCE_NAMES[state]))
        next_state = prev_state[state][end]
        end = start
        state = next_state
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))


def _min_prefix_costs_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    p: int, seed_cost: int = 0,
) -> list[int]:
    """Cost-only prefix DP seeded at ``p`` (``f[p] = seed_cost``)."""
    f = [_INF] * (n + 1)
    f[p] = seed_cost
    windows = (deque([p]), deque([p]))
    for j in range(p + 1, n + 1):
        best = None
        for s, window in enumerate(windows):
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if not window:
                continue
            i = window[0]
            value = f[i] - prefixes[s][i] + prefixes[s][j] + fees[s]
            if best is None or value < best:
                best = value
        if best is None:
            continue
        f[j] = best
        for s, window in enumerate(windows):
            value = f[j] - prefixes[s][j]
            while window:
                tail = window[-1]
                if f[tail] - prefixes[s][tail] <= value:
                    break
                window.pop()
            window.append(j)
    return f


def _certainty_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    p: int, seed_cost: int, optimum: int,
):
    """Per-source coverage difference for minimum-cost completions of
    ``[p, n)``.

    Identical split test as :func:`_certainty`, except the prefix cost table
    is seeded at ``p`` with the fixed prefix's repriced cost: a source-s
    segment ``[i, j)`` belongs to a global optimum iff

        f[i] + fee[s] + P_s[j] - P_s[i] + g[j] == optimum

    with ``f[p]`` equal to the fixed prefix cost and ``g`` unchanged.
    """
    f = _min_prefix_costs_suffix(
        n, prefixes, fees, max_len, availability, p, seed_cost)
    g = _min_suffix_costs(n, prefixes, fees, max_len, availability)

    windows = (deque([p]), deque([p]))
    difference = ([0] * (n + 1), [0] * (n + 1))

    for j in range(p + 1, n + 1):
        for s, window in enumerate(windows):
            low = max(j - max_len, availability.last_down[s][j] + 1)
            while window and window[0] < low:
                window.popleft()
            if not window or g[j] >= _INF:
                continue
            i = window[0]
            target = optimum - g[j] - fees[s] - prefixes[s][j]
            if f[i] - prefixes[s][i] == target:
                diff = difference[s]
                diff[i] += 1
                diff[j] -= 1

        if f[j] < _INF:
            for s, window in enumerate(windows):
                key = f[j] - prefixes[s][j]
                while window and \
                        f[window[-1]] - prefixes[s][window[-1]] > key:
                    window.pop()
                window.append(j)
    return difference


# --------------------------------------------------------------------------
# Capped suffix recurrences: layers c count *total* segments (executed +
# new), so layer c is reached from layer c-1 and only layers
# c0..c0+remaining_budget are swept, where c0 is the executed segment count.
# Indices start at p; layer c0 has the single seed (p, cost_of_prefix) and
# later layers read only endpoints p+1..n-1 of the previous layer.
# --------------------------------------------------------------------------


def _activate_suffix_prefix(window: deque, index: int,
                            previous_cost: array, prefix_s: list[int]):
    """Insert one previous-layer suffix endpoint into a source deque."""
    value = previous_cost[index]
    if value >= _INF_Q:
        return
    key = value - prefix_s[index]
    while window and \
            previous_cost[window[-1]] - prefix_s[window[-1]] > key:
        window.pop()
    window.append(index)


def _capped_min_prefix_costs_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int, remaining_budget: int, p: int,
    seed_cost: int, seed_count: int,
) -> list[array]:
    """Exact-count prefix tables over the suffix, layered by total count.

    ``f[seed_count][p] = seed_cost`` is the only finite seed; layer c
    (``c > seed_count``) activates endpoints ``p+1..n-1`` of layer c-1
    lazily so a layer never reads its own endpoints. Unreachable layers keep
    ``_INF_Q``.
    """
    f = [_q_table(n, _INF_Q) for _ in range(max_segments + 1)]
    f[seed_count][p] = seed_cost

    for d in range(1, remaining_budget + 1):
        c = seed_count + d
        previous = f[c - 1]
        current = f[c]
        windows = (deque(), deque())
        for j in range(p + 1, n + 1):
            predecessor = j - 1
            if d == 1:
                if predecessor == p:
                    for s, window in enumerate(windows):
                        _activate_suffix_prefix(
                            window, p, previous, prefixes[s])
            elif predecessor >= p + 1:
                for s, window in enumerate(windows):
                    _activate_suffix_prefix(
                        window, predecessor, previous, prefixes[s])

            best = _INF_Q
            for s, window in enumerate(windows):
                low = max(j - max_len, availability.last_down[s][j] + 1)
                while window and window[0] < low:
                    window.popleft()
                if window:
                    i = window[0]
                    value = previous[i] - prefixes[s][i] \
                        + prefixes[s][j] + fees[s]
                    if value < best:
                        best = value
            if best < _INF_Q:
                current[j] = best
    return f


def _solve_capped_default_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int, remaining_budget: int, p: int,
    seed_cost: int, seed_count: int,
):
    """Default objective on the suffix under a total-segment cap.

    Inside one layer candidates compare ``(cost, predecessor, source)``; the
    final plan minimizes ``(cost, total layer)``. Backpointers stop at the
    boundary ``p``; the executed prefix is prepended by the caller.
    """
    exact_f = [_q_table(n, _INF_Q) for _ in range(max_segments + 1)]
    exact_f[seed_count][p] = seed_cost
    prev_index = [
        array("i", [-1] * (n + 1)) for _ in range(max_segments + 1)
    ]
    prev_source = [
        array("b", [_NO_SOURCE] * (n + 1))
        for _ in range(max_segments + 1)
    ]

    for d in range(1, remaining_budget + 1):
        c = seed_count + d
        previous = exact_f[c - 1]
        windows = (deque(), deque())
        for j in range(p + 1, n + 1):
            predecessor = j - 1
            if d == 1:
                if predecessor == p:
                    for s, window in enumerate(windows):
                        _activate_suffix_prefix(
                            window, p, previous, prefixes[s])
            elif predecessor >= p + 1 and previous[predecessor] < _INF_Q:
                for s, window in enumerate(windows):
                    _activate_suffix_prefix(
                        window, predecessor, previous, prefixes[s])

            best_candidate = None
            for s, window in enumerate(windows):
                low = max(j - max_len, availability.last_down[s][j] + 1)
                while window and window[0] < low:
                    window.popleft()
                if not window:
                    continue
                i = window[0]
                candidate = (
                    previous[i] - prefixes[s][i]
                    + prefixes[s][j] + fees[s],
                    i,
                    s,
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
            if best_candidate is None:
                continue
            cost, i, s = best_candidate
            exact_f[c][j] = cost
            prev_index[c][j] = i
            prev_source[c][j] = s

    final_layer = -1
    final_cost = _INF_Q
    for d in range(0, remaining_budget + 1):
        c = seed_count + d
        if exact_f[c][n] < final_cost:
            final_cost = exact_f[c][n]
            final_layer = c
    if final_layer < 0:  # pragma: no cover - feasibility prechecked
        raise SegmentLimitExceeded()

    segments = []
    end = n
    layer = final_layer
    while end > p:
        s = prev_source[layer][end]
        start = prev_index[layer][end]
        segments.append(Segment(start, end, SOURCE_NAMES[s]))
        end = start
        layer -= 1
    segments.reverse()
    return Solution(cost=final_cost, segments=tuple(segments)), exact_f


def _certainty_capped_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int, remaining_budget: int, exact_f: list[array],
    p: int, seed_cost: int, seed_count: int, optimum: int,
):
    """Capped certainty difference over minimum-cost completions of
    ``[p, n)``.

    The membership test is the capped split with total-count layers: a
    source-``s`` segment ``[i, j)`` appended to an exact-``c`` prefix is in
    an optimal plan iff

        f[c][i] - P_s[i] + P_s[j] + fee[s] + g[M-c][j] == optimum

    where only ``c >= seed_count`` prefixes exist and the executed boundary
    alone seeds layer ``seed_count``.
    """
    g = _capped_min_suffix_costs(
        n, prefixes, fees, max_len, availability, max_segments)

    windows = tuple(
        (deque(), deque()) for _ in range(max_segments + 1)
    )
    difference = ([0] * (n + 1), [0] * (n + 1))

    for j in range(p + 1, n + 1):
        predecessor = j - 1
        for d in range(1, remaining_budget + 1):
            c = seed_count + d
            if d == 1:
                if predecessor == p and exact_f[c - 1][p] < _INF_Q:
                    for s, window in enumerate(windows[c]):
                        _activate_suffix_prefix(
                            window, p, exact_f[c - 1], prefixes[s])
            elif predecessor >= p + 1 and \
                    exact_f[c - 1][predecessor] < _INF_Q:
                for s, window in enumerate(windows[c]):
                    _activate_suffix_prefix(
                        window, predecessor, exact_f[c - 1], prefixes[s])

            remaining = g[max_segments - c]
            if remaining[j] >= _INF_Q:
                continue
            for s, window in enumerate(windows[c]):
                low = max(j - max_len, availability.last_down[s][j] + 1)
                while window and window[0] < low:
                    window.popleft()
                if not window:
                    continue
                i = window[0]
                target = optimum - remaining[j] - fees[s] - prefixes[s][j]
                if exact_f[c - 1][i] - prefixes[s][i] == target:
                    diff = difference[s]
                    diff[i] += 1
                    diff[j] -= 1
    return difference


def _solve_capped_continuity_suffix(
    n: int, prefixes: tuple[list[int], list[int]],
    fees: tuple[int, int], max_len: int, availability: Availability,
    max_segments: int, remaining_budget: int, p: int,
    seed_cost: int, seed_count: int, seed_switches: int, seed_source: int,
) -> Solution:
    """Continuity objective on the suffix, layered by total segment count.

    The executed prefix enters only layer ``seed_count``, state
    ``seed_source`` at index ``p``; the first new layer (d == 1) builds
    first *new* segments from that one state and counts the boundary switch.
    Later layers read previous-layer A/B states at indices ``p+1..n-1``.
    Final selection compares ``(cost, switches, total layer, last-start,
    final source)``.
    """
    final_candidate = None
    final_state = SOURCE_A
    final_layer = seed_count

    prev_index = [None] * (max_segments + 1)
    prev_state = [None] * (max_segments + 1)

    previous_cost = None
    previous_switches = None
    previous_keys = None

    for d in range(1, remaining_budget + 1):
        c = seed_count + d
        best_cost = [[_INF_Q] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
        switches = [[0] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
        full_keys = [[None] * (n + 1) for _ in (SOURCE_A, SOURCE_B)]
        layer_prev_index = [
            array("i", [-1] * (n + 1)) for _ in (SOURCE_A, SOURCE_B)
        ]
        layer_prev_state = [
            array("b", [_NO_SOURCE] * (n + 1))
            for _ in (SOURCE_A, SOURCE_B)
        ]

        if d == 1:
            # First new segments out of the executed boundary state.
            for s in (SOURCE_A, SOURCE_B):
                window = deque([p])
                for j in range(p + 1, n + 1):
                    low = max(j - max_len,
                              availability.last_down[s][j] + 1)
                    while window and window[0] < low:
                        window.popleft()
                    if not window:
                        continue
                    candidate = (
                        seed_cost - prefixes[s][p] + prefixes[s][j]
                        + fees[s],
                        seed_switches + (0 if seed_source == s else 1),
                        c,
                        p,
                        s,
                        (),
                    )
                    best_cost[s][j] = candidate[0]
                    switches[s][j] = candidate[1]
                    full_keys[s][j] = candidate
                    layer_prev_index[s][j] = p
                    layer_prev_state[s][j] = seed_source
        else:
            windows = tuple(
                tuple(deque() for _s in (SOURCE_A, SOURCE_B))
                for _p in (SOURCE_A, SOURCE_B)
            )
            for j in range(p + 1, n + 1):
                predecessor = j - 1
                if predecessor >= p + 1:
                    for prev_s in (SOURCE_A, SOURCE_B):
                        if previous_cost[prev_s][predecessor] >= _INF_Q:
                            continue
                        for new_s, window in enumerate(windows[prev_s]):
                            tail_key = (
                                previous_cost[prev_s][predecessor]
                                - prefixes[new_s][predecessor],
                                previous_switches[prev_s][predecessor],
                            )
                            while window:
                                tail = window[-1]
                                if (
                                    previous_cost[prev_s][tail]
                                    - prefixes[new_s][tail],
                                    previous_switches[prev_s][tail],
                                ) <= tail_key:
                                    break
                                window.pop()
                            window.append(predecessor)

                for s in (SOURCE_A, SOURCE_B):
                    low = max(j - max_len,
                              availability.last_down[s][j] + 1)
                    best_candidate = None
                    chosen_predecessor = _NO_SOURCE
                    for prev_s in (SOURCE_A, SOURCE_B):
                        window = windows[prev_s][s]
                        while window and window[0] < low:
                            window.popleft()
                        if not window:
                            continue
                        i = window[0]
                        candidate = (
                            previous_cost[prev_s][i] - prefixes[s][i]
                            + prefixes[s][j] + fees[s],
                            previous_switches[prev_s][i]
                            + (0 if prev_s == s else 1),
                            c,
                            i,
                            s,
                            previous_keys[prev_s][i],
                        )
                        if best_candidate is None \
                                or candidate < best_candidate:
                            best_candidate = candidate
                            chosen_predecessor = prev_s
                    if best_candidate is None:
                        continue
                    cost, switch_count, _count, i, _s, _tie = \
                        best_candidate
                    best_cost[s][j] = cost
                    switches[s][j] = switch_count
                    full_keys[s][j] = best_candidate
                    layer_prev_index[s][j] = i
                    layer_prev_state[s][j] = chosen_predecessor

        prev_index[c] = layer_prev_index
        prev_state[c] = layer_prev_state
        previous_cost = best_cost
        previous_switches = switches
        previous_keys = full_keys

        for state in (SOURCE_A, SOURCE_B):
            if best_cost[state][n] >= _INF_Q:
                continue
            candidate = (
                best_cost[state][n], switches[state][n], c,
                layer_prev_index[state][n], state,
            )
            if final_candidate is None or candidate < final_candidate:
                final_candidate = candidate
                final_state = state
                final_layer = c

    if final_candidate is None:  # pragma: no cover - feasibility prechecked
        raise SegmentLimitExceeded()

    total_cost = final_candidate[0]
    segments = []
    end = n
    state = final_state
    layer = final_layer
    while end > p:
        start = prev_index[layer][state][end]
        segments.append(Segment(start, end, SOURCE_NAMES[state]))
        next_state = prev_state[layer][state][end]
        end = start
        layer -= 1
        state = next_state
    segments.reverse()
    return Solution(cost=total_cost, segments=tuple(segments))
