"""Request/response models.

Validation here is what produces HTTP 422 for bad length, out-of-range
values, an illegal L or objective. On any such error FastAPI returns only the
standard ``detail`` payload: no cost and no (partial) segment list.

Optional per-source maintenance windows are validated the same way: each
half-open interval must satisfy ``0 <= start < end <= n``; windows of one
source must be strictly increasing and may neither overlap nor touch (the
end of one must be strictly smaller than the start of the next).

The optional ``max_segments`` field is an integer in ``1..8``. When omitted
the response (and the deterministic adjudication behind it) is exactly the
legacy one; when present the solver only considers plans using at most that
many segments.

The optional ``executed_prefix`` field lists already-repaired segments in
order. They must tile ``[0, p)`` back-to-back (no gaps, no overlaps), each
respecting the length bound ``L`` and its source's maintenance windows; the
service recomputes their cost itself and never accepts a caller-supplied
cost, so the segment objects expose no such field. All of this is validated
up front and rejected with a locatable 422 (loc rooted at
``body.executed_prefix``) before any solver work begins.
"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

SourceName = Literal["A", "B"]
ObjectiveName = Literal["default", "continuity"]
CertaintyName = Literal["A_ONLY", "B_ONLY", "EITHER"]


class PositionCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: StrictInt = Field(ge=0, le=1_000_000)
    b: StrictInt = Field(ge=0, le=1_000_000)


class UnavailableWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Upper bound against n is enforced in the request-level validator.
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(ge=0)


class Unavailable(BaseModel):
    # Unknown source names (e.g. "C") are validation errors, not silently
    # ignored, so malformed availability payloads still return 422.
    model_config = ConfigDict(extra="forbid")

    A: list[UnavailableWindow] = Field(default_factory=list,
                                      max_length=200_000)
    B: list[UnavailableWindow] = Field(default_factory=list,
                                      max_length=200_000)


class ExecutedSegment(BaseModel):
    """One already-repaired half-open segment of the executed prefix.

    No cost/fee field exists: prefix costs are recomputed from ``costs`` and
    the request-level fees, so a caller cannot influence the total by naming
    a price. Extra fields are rejected like every other request object.
    """

    model_config = ConfigDict(extra="forbid")

    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(ge=0)
    source: SourceName


class GapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: StrictInt = Field(ge=1, le=200_000)
    L: StrictInt = Field(ge=1, le=4_096)
    fee_a: StrictInt = Field(ge=0, le=1_000_000)
    fee_b: StrictInt = Field(ge=0, le=1_000_000)
    objective: ObjectiveName = "default"
    # Optional review-desk limit: only plans using at most this many segments
    # are considered. Omitted/None means no limit and byte-identical legacy
    # behaviour; 1..8 inclusive are the only accepted values.
    max_segments: StrictInt | None = Field(default=None, ge=1, le=8)
    costs: list[PositionCost] = Field(min_length=1, max_length=200_000)
    unavailable: Unavailable | None = None
    # Optional already-executed prefix: ordered half-open segments tiling
    # [0, p) for some 0 <= p <= n. Omitted/None/[] mean no executed work and
    # byte-identical legacy behaviour. Cross-segment structure, the length
    # bound and per-source availability are checked in the field validator.
    executed_prefix: list[ExecutedSegment] | None = Field(
        default=None, max_length=200_000)

    @model_validator(mode="after")
    def _costs_and_windows_must_be_valid(self) -> "GapRequest":
        if len(self.costs) != self.n:
            raise ValueError("len(costs) must equal n")
        if self.unavailable is not None:
            for name in ("A", "B"):
                previous_end = -1
                for window in getattr(self.unavailable, name):
                    if not (0 <= window.start < window.end <= self.n):
                        raise ValueError(
                            f"unavailable.{name} windows must satisfy "
                            "0 <= start < end <= n")
                    # Strictly increasing with a gap: touching (previous end
                    # == this start) is rejected together with overlaps.
                    if window.start <= previous_end:
                        raise ValueError(
                            f"unavailable.{name} windows must be strictly "
                            "increasing and must not overlap or touch")
                    previous_end = window.end
        return self

    @field_validator("executed_prefix")
    @classmethod
    def _executed_prefix_must_tile_legally(cls, segments, info):
        # Declared after them, the already-validated n/L/unavailable values
        # are available here; costs itself is validated by the model
        # validator afterwards.
        if not segments:
            return segments
        data = info.data
        # Sibling fields validate first; when n/L themselves failed they are
        # absent here. Skip the cross-field checks then -- their own errors
        # are already reported and a KeyError must never become a 500.
        if "n" not in data or "L" not in data:
            return segments
        n = data["n"]
        max_len = data["L"]
        unavailable = data.get("unavailable")

        blocked: dict[str, frozenset[int]] = {"A": frozenset(),
                                              "B": frozenset()}
        if unavailable is not None:
            for name in ("A", "B"):
                down = set()
                for window in getattr(unavailable, name):
                    down.update(range(window.start, window.end))
                blocked[name] = frozenset(down)

        errors: list[InitErrorDetails] = []
        cursor = 0
        for index, segment in enumerate(segments):
            within_range = 0 <= segment.start < segment.end <= n

            # Tiling: the first segment starts at 0 and every later segment
            # starts exactly where the previous one ended, so gaps and
            # overlaps are both rejected by the single cursor check.
            if segment.start != cursor:
                errors.append(InitErrorDetails(
                    type=PydanticCustomError(
                        "value_error.executed_prefix_tiling",
                        "executed_prefix segments must be ordered and cover "
                        "[0, p) back-to-back with no gaps or overlaps: "
                        f"segment {index} starts at {segment.start}, "
                        f"expected {cursor}"),
                    loc=(index, "start"),
                    input=segment.start,
                ))

            if not within_range:
                errors.append(InitErrorDetails(
                    type=PydanticCustomError(
                        "value_error.executed_prefix_range",
                        f"executed_prefix segment {index} must satisfy "
                        f"0 <= start < end <= n (n={n})"),
                    loc=(index, "end"),
                    input=segment.end,
                ))
            elif not (1 <= segment.end - segment.start <= max_len):
                errors.append(InitErrorDetails(
                    type=PydanticCustomError(
                        "value_error.executed_prefix_length",
                        f"executed_prefix segment {index} length "
                        f"{segment.end - segment.start} must satisfy "
                        f"1 <= length <= L ({max_len})"),
                    loc=(index, "end"),
                    input=segment.end,
                ))
            elif any(position in blocked[segment.source]
                     for position in range(segment.start, segment.end)):
                errors.append(InitErrorDetails(
                    type=PydanticCustomError(
                        "value_error.executed_prefix_availability",
                        f"executed_prefix segment {index} uses source "
                        f"{segment.source} inside that source's maintenance "
                        "window"),
                    loc=(index, "source"),
                    input=segment.source,
                ))

            # Advance along the declared segment so later diagnostics stay
            # meaningful even after one bad boundary.
            cursor = max(cursor, segment.end)

        if errors:
            raise ValidationError.from_exception_data(
                cls.__name__, errors)
        return segments


class SegmentOut(BaseModel):
    start: int
    end: int
    source: SourceName


class GapResponse(BaseModel):
    cost: int
    segments: list[SegmentOut]
    # Per position 0..n-1: A_ONLY / B_ONLY mean every minimum-cost cover
    # uses that source there; EITHER means both sources occur in at least one
    # minimum-cost cover. Objective-independent.
    certainty: list[CertaintyName]
