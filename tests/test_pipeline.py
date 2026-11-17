import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig

def test_pipeline_end_to_end(tmp_path):
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=200, chunk_overlap=30)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=3)
    pipe = RAGPipeline(cfg)
    txt = "Machine learning is a subset of AI. Deep learning uses neural networks. RAG combines retrieval and generation."
    pipe.ingest_text(txt, metadata={"source": "t"})
    assert len(pipe.chunks) >= 1
    res = pipe.query("What is RAG?", top_k=2)
    assert "answer" in res
    assert len(res["retrieved"]) >= 1
    assert pipe.graph_stats()["num_entities"] >= 0

def test_persistence(tmp_path):
    cfg = RAGConfig()
    pipe = RAGPipeline(cfg)
    pipe.ingest_text("Hello world. This is a test document.", metadata={"source": "t"})
    d = str(tmp_path / "idx")
    pipe.save(d)
    p2 = RAGPipeline.load(d)
    assert len(p2.chunks) == len(pipe.chunks)