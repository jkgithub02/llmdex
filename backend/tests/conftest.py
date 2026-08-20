"""Suite-wide guards.

``ALLOW_MODEL_REQUESTS = False`` makes pydantic-ai raise if anything but a test
model is called. The offline suite already avoids the network by injecting
fixtures; this makes a slip loud rather than expensive, since a real call here
costs money and reaches a live endpoint.

The live tier re-enables it for its own duration -- see ``backend/tests/e2e``.
"""

import pytest
from pydantic_ai import models


@pytest.fixture(autouse=True)
def _no_real_model_requests():
    models.ALLOW_MODEL_REQUESTS = False
    yield
