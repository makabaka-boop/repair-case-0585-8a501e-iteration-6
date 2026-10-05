"""Strict parsing and validation for ``POST /solve`` request bodies.

Pydantic rejects fields that are not declared on a model. JSON itself, however,
permits duplicate keys in an object and conventional parsers silently keep the
last value. Detecting those keys while the raw document is parsed makes both
classes of errors stable 422 validation failures before any solver work begins.
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
        return GapRequest.model_validate(raw)
    except ValidationError as exc:
        raise RequestValidationError(_with_body_location(exc.errors()))
