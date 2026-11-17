"""FastAPI service for the RAG pipeline."""
from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig, GenerationConfig
from rag.pipeline import RAGPipeline

app = FastAPI(title="RAG API", version="1.0")

# Global pipeline (initialized on startup or via /init)
PIPELINE: Optional[RAGPipeline] = None


class InitRequest(BaseModel):
    data_dir: Optional[str] = None
    text: Optional[str] = None
    chunking_strategy: str = "recursive"
    chunk_size: int = 400
    retrieval_mode: str = "hybrid"
    top_k: int = 5


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    generate: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    contexts: List[str] = []
    chunk_ids: List[str] = []
    retrieved: List[dict] = []


@app.get("/health")
def health():
    return {"status": "ok", "ready": PIPELINE is not None}


@app.post("/init")
def init(req: InitRequest):
    global PIPELINE
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy=req.chunking_strategy, chunk_size=req.chunk_size)
    cfg.retrieval = RetrievalConfig(mode=req.retrieval_mode, top_k=req.top_k)
    pipe = RAGPipeline(cfg)
    if req.data_dir:
        pipe.ingest(req.data_dir, chunking_strategy=req.chunking_strategy)
    if req.text:
        pipe.ingest_text(req.text)
    PIPELINE = pipe
    return {"status": "ok", "chunks": len(pipe.chunks), "graph": pipe.graph_stats()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global PIPELINE
    if PIPELINE is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized. Call /init first.")
    res = PIPELINE.query(req.query, top_k=req.top_k, generate=req.generate)
    retrieved = [{"chunk_id": c.chunk_id, "score": float(s), "text": c.text[:300]} for c, s in res.get("retrieved", [])]
    return QueryResponse(
        query=res["query"],
        answer=res.get("answer"),
        contexts=res.get("contexts", []),
        chunk_ids=res.get("chunk_ids", []),
        retrieved=retrieved,
    )


@app.get("/graph/stats")
def graph_stats():
    if PIPELINE is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized.")
    return PIPELINE.graph_stats()


@app.post("/agentic")
def agentic(req: QueryRequest):
    global PIPELINE
    if PIPELINE is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized.")
    from rag.advanced.agentic_rag import AgenticRAG
    agent = AgenticRAG(PIPELINE, use_hyde=True, max_iters=1)
    res = agent.run(req.query, top_k=req.top_k or 5)
    return {
        "query": res["query"],
        "answer": res["answer"],
        "confidence": res["confidence"],
        "conflicts": len(res["conflicts"]),
        "chunk_ids": res["chunk_ids"],
    }


@app.post("/debate")
def debate(req: QueryRequest):
    global PIPELINE
    if PIPELINE is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized.")
    from rag.advanced.multi_agent_debate import MultiAgentDebateRAG
    d = MultiAgentDebateRAG(PIPELINE)
    res = d.run(req.query, top_k=req.top_k or 5)
    return {"query": res["query"], "answer": res["answer"], "winner": res["winner"], "scores": res["scores"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)