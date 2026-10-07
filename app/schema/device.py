"""Pydantic data contracts for the API."""
from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class Device(BaseModel):
    brand: str
    model: str
    specs: Dict[str, str]
    release_year: int
    price_tier: Literal["budget", "mid-range", "flagship"]


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Messy mobile-device text")



class ExtractResponse(Device):
    pass


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, description="Search query (max 500 chars)")


class SearchResult(BaseModel):
    document: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
