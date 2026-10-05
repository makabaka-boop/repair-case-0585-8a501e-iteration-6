"""Acceptance tests for strict request-schema handling.

The tests use raw JSON documents because Python dictionaries (and therefore
TestClient's ``json=`` helper) cannot represent duplicate object keys. The
control instance is chosen so omitting the maintenance window changes both the
winning source and the returned plan: without it A covers the gap for cost 0;
when A is down at position 0, B must cover that position.
"""

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BASE_FIELDS = (
    '"n": 2, '
    '"L": 2, '
    '"fee_a": 0, '
    '"fee_b": 0, '
    '"costs": [{"a": 0, "b": 100}, {"a": 0, "b": 100}]'
)
RESTRICTION = '"unavailable": {"A": [{"start": 0, "end": 1}]}'

TIE_FIELDS = (
    '"n": 3, '
    '"L": 1, '
    '"fee_a": 0, '
    '"fee_b": 0, '
    '"costs": [{"a": 0, "b": 0}, {"a": 1, "b": 0}, '
    '{"a": 0, "b": 1}]'
)


def raw_document(inner):
    return "{" + inner + "}"


def post_raw(document):
    return client.post(
        "/solve",
        content=document,
        headers={"Content-Type": "application/json"},
    )


def control_payload(**overrides):
    payload = {
        "n": 2,
        "L": 2,
        "fee_a": 0,
        "fee_b": 0,
        "costs": [{"a": 0, "b": 100}, {"a": 0, "b": 100}],
    }
    payload.update(overrides)
    return payload


def assert_detail_only_422(response):
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"detail"}
    assert isinstance(body["detail"], list)
    assert body["detail"]
    for error in body["detail"]:
        assert "loc" in error
        assert "msg" in error
        assert "type" in error
        assert "cost" not in error
        assert "segments" not in error
        assert "certainty" not in error


def assert_has_location(response, location):
    locations = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert location in locations


def test_control_data_changes_optimal_source_without_restriction():
    unrestricted = client.post("/solve", json=control_payload())
    restricted = client.post(
        "/solve", json=control_payload(unavailable={
            "A": [{"start": 0, "end": 1}],
        }))

    assert unrestricted.status_code == restricted.status_code == 200
    assert unrestricted.json() == {
        "cost": 0,
        "segments": [{"start": 0, "end": 2, "source": "A"}],
        "certainty": ["A_ONLY", "A_ONLY"],
    }
    assert restricted.json() == {
        "cost": 100,
        "segments": [
            {"start": 0, "end": 1, "source": "B"},
            {"start": 1, "end": 2, "source": "A"},
        ],
        "certainty": ["B_ONLY", "A_ONLY"],
    }


def test_legacy_and_compatible_requests_remain_unchanged():
    baseline = client.post("/solve", json=control_payload())
    omitted_objective = client.post("/solve", json=control_payload())
    explicit_default = client.post(
        "/solve", json=control_payload(objective="default"))
    null_unavailable = client.post(
        "/solve", json=control_payload(unavailable=None))
    empty_unavailable = client.post(
        "/solve", json=control_payload(unavailable={}))
    empty_lists = client.post(
        "/solve", json=control_payload(unavailable={"A": [], "B": []}))

    assert baseline.status_code == 200
    assert omitted_objective.json() == explicit_default.json()
    assert null_unavailable.json() == baseline.json()
    assert empty_unavailable.json() == baseline.json()
    assert empty_lists.json() == baseline.json()


def test_both_established_objectives_remain_valid():
    payload = {
        "n": 3,
        "L": 1,
        "fee_a": 0,
        "fee_b": 0,
        "costs": [
            {"a": 0, "b": 0},
            {"a": 1, "b": 0},
            {"a": 0, "b": 1},
        ],
    }
    default_response = client.post(
        "/solve", json={**payload, "objective": "default"})
    continuity_response = client.post(
        "/solve", json={**payload, "objective": "continuity"})

    assert default_response.status_code == continuity_response.status_code == 200
    assert default_response.json()["segments"] == [
        {"start": 0, "end": 1, "source": "A"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
    assert continuity_response.json()["segments"] == [
        {"start": 0, "end": 1, "source": "B"},
        {"start": 1, "end": 2, "source": "B"},
        {"start": 2, "end": 3, "source": "A"},
    ]
    assert default_response.json()["cost"] == \
        continuity_response.json()["cost"] == 0
    assert default_response.json()["certainty"] == \
        continuity_response.json()["certainty"]


INVALID_DOCUMENTS = [
    # Top level: typo, extra key, duplicate objective/unavailable.
    (
        "top-level-unavailable-typo",
        ("body", "unavailble"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailble": {"A": [{"start": 0, "end": 1}]}'),
    ),
    (
        "top-level-objective-typo",
        ("body", "objectve"),
        raw_document(f'{TIE_FIELDS}, "objectve": "continuity"'),
    ),
    (
        "top-level-extra-key",
        ("body", "extension"),
        raw_document(f'{BASE_FIELDS}, {RESTRICTION}, "extension": true'),
    ),
    (
        "top-level-duplicate-objective-default-last",
        ("body", "objective"),
        raw_document(
            f'{TIE_FIELDS}, '
            '"objective": "continuity", "objective": "default"'),
    ),
    (
        "top-level-duplicate-objective-continuity-last",
        ("body", "objective"),
        raw_document(
            f'{TIE_FIELDS}, '
            '"objective": "default", "objective": "continuity"'),
    ),
    (
        "top-level-duplicate-unavailable-null-last",
        ("body", "unavailable"),
        raw_document(
            f'{BASE_FIELDS}, {RESTRICTION}, "unavailable": null'),
    ),
    (
        "top-level-duplicate-unavailable-window-last",
        ("body", "unavailable"),
        raw_document(
            f'{BASE_FIELDS}, "unavailable": null, {RESTRICTION}'),
    ),

    # Cost object: typo, extra key, duplicate cost key.
    (
        "cost-key-typo",
        ("body", "costs", 0, "aa"),
        raw_document(
            f'"n": 2, "L": 2, "fee_a": 0, "fee_b": 0, '
            '"costs": [{"a": 0, "aa": 100, "b": 100}, '
            '{"a": 0, "b": 100}], '
            f'{RESTRICTION}'),
    ),
    (
        "cost-extra-key",
        ("body", "costs", 0, "extension"),
        raw_document(
            f'"n": 2, "L": 2, "fee_a": 0, "fee_b": 0, '
            '"costs": [{"a": 0, "b": 100, "extension": 1}, '
            '{"a": 0, "b": 100}], '
            f'{RESTRICTION}'),
    ),
    (
        "cost-duplicate-a-expensive-last",
        ("body", "costs", 1, "a"),
        raw_document(
            f'"n": 2, "L": 2, "fee_a": 0, "fee_b": 0, '
            '"costs": [{"a": 0, "b": 100}, '
            '{"a": 0, "a": 100, "b": 100}], '
            f'{RESTRICTION}'),
    ),
    (
        "cost-duplicate-a-zero-last",
        ("body", "costs", 1, "a"),
        raw_document(
            f'"n": 2, "L": 2, "fee_a": 0, "fee_b": 0, '
            '"costs": [{"a": 0, "b": 100}, '
            '{"a": 100, "a": 0, "b": 100}], '
            f'{RESTRICTION}'),
    ),

    # Unavailable object: typo/unknown source, extra source, duplicate source.
    (
        "unavailable-source-typo",
        ("body", "unavailable", "C"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"C": [{"start": 0, "end": 1}]}'),
    ),
    (
        "unavailable-extra-source",
        ("body", "unavailable", "C"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 0, "end": 1}], '
            '"C": []}'),
    ),
    (
        "unavailable-duplicate-source-empty-last",
        ("body", "unavailable", "A"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 0, "end": 1}], "A": []}'),
    ),
    (
        "unavailable-duplicate-source-window-last",
        ("body", "unavailable", "A"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [], "A": [{"start": 0, "end": 1}]}'),
    ),

    # Window object: typo, extra key, duplicate boundaries.
    (
        "window-start-typo",
        ("body", "unavailable", "A", 0, "stat"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 0, "stat": 1, "end": 1}]}'),
    ),
    (
        "window-extra-key",
        ("body", "unavailable", "A", 0, "extension"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 0, "end": 1, '
            '"extension": true}]}'),
    ),
    (
        "window-duplicate-start-one-last",
        ("body", "unavailable", "A", 0, "start"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 0, "start": 1, '
            '"end": 2}]}'),
    ),
    (
        "window-duplicate-start-zero-last",
        ("body", "unavailable", "A", 0, "start"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 1, "start": 0, '
            '"end": 2}]}'),
    ),
    (
        "window-duplicate-end-two-last",
        ("body", "unavailable", "A", 0, "end"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 0, "end": 1, '
            '"end": 2}]}'),
    ),
    (
        "window-duplicate-end-one-last",
        ("body", "unavailable", "A", 0, "end"),
        raw_document(
            f'{BASE_FIELDS}, '
            '"unavailable": {"A": [{"start": 0, "end": 2, '
            '"end": 1}]}'),
    ),
]


def test_invalid_document_cases_are_well_formed_json():
    # Guardrail: a 422 must be caused by schema/duplicate-key rejection, not
    # by a typo accidentally introduced into the acceptance JSON itself.
    for _name, _location, document in INVALID_DOCUMENTS:
        json.loads(document)


def test_malformed_json_is_detail_only_422():
    response = post_raw(raw_document(f'{BASE_FIELDS},'))
    assert_detail_only_422(response)
    assert response.json()["detail"][0]["loc"][0] == "body"


def test_non_object_json_is_detail_only_422():
    response = post_raw("[]")
    assert_detail_only_422(response)
    assert_has_location(response, ("body",))


def test_empty_body_is_detail_only_422_without_partial_solution():
    response = client.post(
        "/solve",
        content=b"",
        headers={"Content-Type": "application/json"},
    )
    assert_detail_only_422(response)
    assert "cost" not in response.json()


def test_invalid_documents_are_rejected_before_scheduling():
    for name, location, document in INVALID_DOCUMENTS:
        response = post_raw(document)
        assert_detail_only_422(response)
        assert_has_location(response, location), name


def test_openapi_documents_strict_request_body():
    schema = app.openapi()
    request_schema = schema["paths"]["/solve"]["post"]["requestBody"][
        "content"]["application/json"]["schema"]

    assert request_schema == {"$ref": "#/components/schemas/GapRequest"}
    gap_schema = schema["components"]["schemas"]["GapRequest"]
    assert gap_schema["additionalProperties"] is False
    assert schema["components"]["schemas"]["PositionCost"][
        "additionalProperties"] is False
    assert schema["components"]["schemas"]["Unavailable"][
        "additionalProperties"] is False
    assert schema["components"]["schemas"]["UnavailableWindow"][
        "additionalProperties"] is False
