"""FastAPI dependencies shared by every feature router.

``get_store`` is a function rather than a module-level value so tests can replace
it through ``app.dependency_overrides`` -- which is how the read endpoints are
exercised with the network removed entirely (R7.2).

Nothing feature-specific belongs here. A dependency only one feature uses lives
with that feature, so ``core`` never imports a feature and the dependency
direction stays one-way.
"""

from typing import Annotated

from fastapi import Depends

from backend.core.config import store_from_env
from backend.core.store import Store


def get_store() -> Store:
    return store_from_env()


StoreDep = Annotated[Store, Depends(get_store)]
