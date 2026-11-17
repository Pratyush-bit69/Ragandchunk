import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
from rag.advanced.agentic_rag import AgenticRAG
from rag.advanced.query_decomposition import RuleBasedDecomposer
from rag.advanced.conflict import ConflictDetector
from rag.advanced.confidence import ConfidenceEstimator
from rag.advanced.late_interaction import LateInteractionRetriever
from rag.advanced.complex_evaluation import abstention_quality, provenance_score

def test_decomposition():
    d = RuleBasedDecomposer()
    subs = d.decompose("Compare supervised and unsupervised learning")
    assert len(subs) >= 2

def test_agentic(tmp_path):
    cfg = RAGConfig()
    cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=200, chunk_overlap=30)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=3)
    pipe = RAGPipeline(cfg)
    pipe.ingest_text("Machine learning includes supervised and unsupervised learning. Supervised uses labels. Unsupervised finds patterns.", metadata={"source": "t"})
    agent = AgenticRAG(pipe, use_hyde=False, max_iters=1)
    res = agent.run("What are types of machine learning?", top_k=2)
    assert "answer" in res
    assert "confidence" in res
    assert "provenance" in res

def test_conflict():
    from rag.chunking import Chunk
    chunks = [Chunk(text="X is true.", chunk_id="a"), Chunk(text="X is not true.", chunk_id="b")]
    cd = ConflictDetector()
    cfs = cd.detect(chunks)
    assert len(cfs) >= 1

def test_confidence():
    from rag.chunking import Chunk
    ce = ConfidenceEstimator(abstention_threshold=0.4)
    chunks = [Chunk(text="RAG combines retrieval and generation.", chunk_id="a")]
    r = ce.score("What is RAG?", "RAG combines retrieval and generation.", chunks, [0.8])
    assert "abstain" in r
    assert r["overall"] > 0

def test_late_interaction():
    from rag.chunking import Chunk
    chunks = [Chunk(text="RAG combines retrieval and generation.", chunk_id="a"), Chunk(text="Machine learning is AI.", chunk_id="b")]
    pipe = None
    # need embeddings
    from rag.embeddings import HashingEmbeddings
    emb = HashingEmbeddings(dim=64)
    li = LateInteractionRetriever(emb, chunks)
    res = li.retrieve("What is RAG?", k=1)
    assert len(res) == 1

def test_abstention_quality():
    preds = [{"abstained": True}, {"abstained": False}]
    labels = [True, False]
    r = abstention_quality(preds, labels)
    assert r["accuracy"] == 1.0

def test_provenance():
    prov = [{"supported": True}, {"supported": False}]
    assert provenance_score(prov) == 0.5