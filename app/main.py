"""FastAPI app + all routes (HTTP concerns only)."""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()  

from app.schema.device import (  
    ExtractRequest,
    ExtractResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.extraction import ExtractionError, extract_device  # noqa: E402
from app.services.search import hybrid_search  
from app.utils.security import is_suspicious  

MAX_QUERY_LENGTH = 500
MAX_EXTRACT_LENGTH = 5000


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    from app.utils.bm25_index import get_bm25_index
    from app.utils.embeddings import get_doc_embeddings

    get_bm25_index()
    get_doc_embeddings()
    yield


app = FastAPI(title="Mobile Device Assistant API", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    if len(req.text) > MAX_EXTRACT_LENGTH:
        raise HTTPException(400, f"Text longer than {MAX_EXTRACT_LENGTH} characters")
    if is_suspicious(req.text):
        raise HTTPException(400, "Input rejected: suspicious content detected")
    try:
        return extract_device(req.text)
    except ExtractionError as exc:
        raise HTTPException(502, str(exc))
    except RuntimeError as exc: 
        raise HTTPException(500, str(exc))


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if len(req.q) > MAX_QUERY_LENGTH:
        raise HTTPException(400, f"Query longer than {MAX_QUERY_LENGTH} characters")
    results = hybrid_search(req.q, k=3)
    return SearchResponse(
        results=[SearchResult(document=doc, score=score) for doc, score in results]
    )
