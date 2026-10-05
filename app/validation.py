"""Strict parsing and validation for ``POST /solve`` request bodies.

Pydantic rejects fields that are not declared on a model. JSON itself, however,
permits duplicate keys in an object and conventional parsers silently keep the
last value. Detecting those keys while the raw document is parsed makes both
classes of errors stable 422 validation failures before any solver work begins.

The optional executed prefix also carries cross-field constraints that the
plain schema cannot express: the ordered half-open segments must tile
``[0, p)`` with no gaps or overlaps, respect ``L`` and stay clear of their own
source's maintenance windows. Those checks run here, after schema parsing, and
emit locatable ``body.prefix...`` 422 errors identical in shape to the rest.
"""

import json

from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from .models import GapRequest


class _DuplicateTrackingObject(dict):
    """Dict preserving the names of keys repeated in one JSON object."""

    def __init__(self, pairs):
        super().__init__(pairs)
        seen = set()
        duplicate_keys = []
        for key, _value in pairs:
            if key in seen and key not in duplicate_keys:
                duplicate_keys.append(key)
            seen.add(key)
        self.duplicate_keys = tuple(duplicate_keys)


def _collect_duplicate_locations(value, path=()):
    errors = []
    if isinstance(value, dict):
        for key in getattr(value, "duplicate_keys", ()):
            errors.append(path + (key,))
        for key, item in value.items():
            errors.extend(
                _collect_duplicate_locations(item, path + (str(key),)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(
                _collect_duplicate_locations(item, path + (index,)))
    return errors


def _with_body_location(errors):
    normalized = []
    for error in errors:
        loc = tuple(error.get("loc", ()))
        if not loc or loc[0] != "body":
            loc = ("body",) + loc
        normalized.append({**error, "loc": loc})
    return normalized


def _prefix_error(index, message, field=None):
    loc = ("body", "prefix")
    if index is not None:
        loc += (index,)
    if field is not None:
        loc += (field,)
    return {
        "type": "value_error.executed_prefix",
        "loc": loc,
        "msg": message,
        "input": field if field is not None else "prefix",
    }


def validate_executed_prefix(request: GapRequest) -> None:
    """Validate the executed prefix against L, n and the maintenance windows.

    Every failure is a locatable 422 -- the request is malformed, not an
    unsolvable instance -- so an illegal prefix never reaches the solver and
    can never produce a partial plan.
    """
    segments = request.prefix
    if segments is None:
        return

    # Per-source blocked positions from the validated half-open windows, via
    # the same O(n + m) difference sweep as the solver: intersecting a fixed
    # segment must not degrade into a segment-times-windows scan.
    blocked_by_source: dict[str, bytearray] = {}
    if request.unavailable is not None:
        for source_name, windows in (
                ("A", request.unavailable.A),
                ("B", request.unavailable.B)):
            mark = [0] * (request.n + 1)
            for window in windows:
                mark[window.start] += 1
                mark[window.end] -= 1
            flags = bytearray(request.n)
            active = 0
            for k in range(request.n):
                active += mark[k]
                flags[k] = 1 if active > 0 else 0
            blocked_by_source[source_name] = flags

    errors = []
    expected_start = 0
    for index, seg in enumerate(segments):
        if index == 0 and seg.start != 0:
            errors.append(_prefix_error(
                index, "executed prefix must start at position 0", "start"))
        if index > 0 and seg.start != expected_start:
            # The previous segment either did not end here (gap) or ends past
            # it (overlap); both are rejected, whichever order the caller used.
            errors.append(_prefix_error(
                index,
                "executed prefix segments must tile [0, p) with no gaps or "
                "overlaps",
                "start"))
        if seg.end <= seg.start:
            errors.append(_prefix_error(
                index, "executed prefix segment must have end > start", "end"))
        if seg.end > request.n:
            errors.append(_prefix_error(
                index,
                "executed prefix segment must end at or before n", "end"))
        if 1 <= seg.end - seg.start and seg.end - seg.start > request.L:
            errors.append(_prefix_error(
                index,
                "executed prefix segment must not be longer than L", "end"))
        expected_start = seg.end

        flags = blocked_by_source.get(seg.source)
        if flags is not None and 0 <= seg.start < seg.end <= request.n \
                and any(flags[k] for k in range(seg.start, seg.end)):
            errors.append(_prefix_error(
                index,
                f"executed prefix segment of source {seg.source} "
                "intersects one of that source's maintenance windows"))

    if errors:
        raise RequestValidationError(errors)


def parse_gap_request_body(content: bytes) -> GapRequest:
    """Parse strict JSON, reject duplicate keys, then validate the model."""
    if not content:
        raw = {}
    else:
        try:
            raw = json.loads(
                content, object_pairs_hook=_DuplicateTrackingObject)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            location = getattr(exc, "pos", None)
            raise RequestValidationError([{
                "type": "json_invalid",
                "loc": ("body",) + (() if location is None
                                    else (location,)),
                "msg": f"JSON decode error: {exc}",
                "input": content.decode(errors="replace"),
            }])

    duplicate_errors = [
        {
            "type": "value_error.duplicate_key",
            "loc": ("body",) + location,
            "msg": "JSON object contains a duplicate key",
            "input": location[-1],
        }
        for location in _collect_duplicate_locations(raw)
    ]
    if duplicate_errors:
        raise RequestValidationError(duplicate_errors)

    try:
        request = GapRequest.model_validate(raw)
    except ValidationError as exc:
        raise RequestValidationError(_with_body_location(exc.errors()))
    validate_executed_prefix(request)
    return request
