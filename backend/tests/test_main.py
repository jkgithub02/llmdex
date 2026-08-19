"""Route order in main.py is load-bearing.

`GET /models/{model_id:path}` is a greedy catch-all. Registered before the
agents router it swallows `/models/{id}/agents/stream` and 404s the stream
before that route is ever tried. A comment says so; this asserts it.
"""

from app.main import app


def _paths() -> list[str]:
    """Flatten app.routes.

    Starlette wraps each ``include_router`` call in an ``_IncludedRouter`` that
    holds its routes on ``original_router`` rather than exposing them directly
    on the app, so a plain ``route.path`` scan sees only the routes declared on
    ``app`` itself (``/health`` and friends) and none of main.py's routers.
    """
    paths: list[str] = []
    for route in app.routes:
        if hasattr(route, "path"):
            paths.append(route.path)
        elif hasattr(route, "original_router"):
            paths.extend(r.path for r in route.original_router.routes if hasattr(r, "path"))
    return paths


def test_the_agent_stream_is_registered_before_the_greedy_model_route():
    paths = _paths()
    assert paths.index("/models/{model_id:path}/agents/stream") < paths.index(
        "/models/{model_id:path}"
    )
