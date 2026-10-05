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
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

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
