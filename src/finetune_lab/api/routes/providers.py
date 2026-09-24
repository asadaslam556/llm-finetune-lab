"""Provider listing. Feeds the chat panel's provider picker."""

from fastapi import APIRouter

from ...providers.registry import list_providers
from ..schemas import ProviderInfo

router = APIRouter(tags=["providers"])


@router.get("/api/providers", response_model=list[ProviderInfo])
def providers():
    return list_providers()
