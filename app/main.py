"""FastAPI application for the telemetry-gap repair DP."""

from fastapi import Depends, FastAPI, HTTPException, Request

from .models import GapRequest, GapResponse
from .solver import NoFeasibleRepair, SegmentLimitExceeded, solve
from .validation import parse_gap_request_body

app = FastAPI(
    title="Telemetry Gap Repair",
    summary="Minimum-cost cover of a telemetry gap by A/B repair segments",
)


def custom_openapi():
    """Expose GapRequest as a normal request-body schema in the OpenAPI spec.

    Parsing uses a dependency so duplicate JSON keys can be rejected from the
    raw bytes; FastAPI would otherwise omit the request body from generated
    docs because that dependency does not declare a model body parameter.
    """
    if app.openapi_schema is not None:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        routes=app.routes,
    )
    request_schemas = GapRequest.model_json_schema(
        ref_template="#/components/schemas/{model}")
    components = schema.setdefault("components", {}).setdefault(
        "schemas", {})
    components.update(request_schemas.pop("$defs", {}))
    components["GapRequest"] = request_schemas
    schema["paths"]["/solve"]["post"]["requestBody"]["content"][
        "application/json"]["schema"] = {
        "$ref": "#/components/schemas/GapRequest"
    }
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


async def validated_gap_request(request: Request) -> GapRequest:
    return parse_gap_request_body(await request.body())


@app.post(
    "/solve",
    response_model=GapResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    # custom_openapi replaces this placeholder with a
                    # component reference so nested models are documented too.
                    "schema": {},
                },
            },
        },
    },
)
def solve_gap(
    req: GapRequest = Depends(validated_gap_request),
) -> GapResponse:
    if req.unavailable is None:
        unavailable = None
    else:
        unavailable = (
            [(w.start, w.end) for w in req.unavailable.A],
            [(w.start, w.end) for w in req.unavailable.B],
        )
    try:
        result = solve(
            n=req.n,
            costs=[(p.a, p.b) for p in req.costs],
            fee_a=req.fee_a,
            fee_b=req.fee_b,
            max_len=req.L,
            objective=req.objective,
            unavailable=unavailable,
            max_segments=req.max_segments,
        )
    except SegmentLimitExceeded:
        # A legal cover exists but its minimum segment count is above the
        # review desk's cap: a distinct, explicit reason. Still detail-only;
        # no cost, segments or certainty may leak from a partial plan.
        raise HTTPException(
            status_code=409, detail="MAX_SEGMENTS_EXCEEDED")
    except NoFeasibleRepair:
        # Only the standard detail field: no cost, segments or certainty may
        # leak from a partial solution.
        raise HTTPException(status_code=409, detail="NO_FEASIBLE_REPAIR")
    return GapResponse(
        cost=result.cost,
        segments=[
            {"start": seg.start, "end": seg.end, "source": seg.source}
            for seg in result.segments
        ],
        certainty=list(result.certainty),
    )
