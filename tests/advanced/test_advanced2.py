import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
from rag.advanced.verified_generation import VerifiedGenerator
from rag.advanced.multi_agent_debate import MultiAgentDebateRAG
from rag.advanced.self_consistency import SelfConsistencyRAG
from rag.advanced.adaptive_retrieval import ComplexityEstimator, AdaptiveRetriever
from rag.advanced.graph_of_thoughts import GraphOfThoughts
from rag.advanced.query_decomposition import RuleBasedDecomposer

def _pipe():
    cfg = RAGConfig(); cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=200, chunk_overlap=30)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=3)
    p = RAGPipeline(cfg); p.ingest_text("Machine learning includes supervised and unsupervised learning. RAG combines retrieval and generation. Evaluation uses precision and recall.", metadata={"source":"t"})
    return p

def test_verified():
    p = _pipe()
    vg = VerifiedGenerator()
    r = vg.generate("What is RAG?", [c for c,_ in p.retrieve("RAG", top_k=2)])
    assert "answer" in r

def test_debate():
    p = _pipe()
    d = MultiAgentDebateRAG(p)
    r = d.run("What is RAG?", top_k=2)
    assert "answer" in r and "winner" in r

def test_self_consistency():
    p = _pipe()
    sc = SelfConsistencyRAG(p, n_samples=3, top_k=2)
    r = sc.run("What is RAG?")
    assert "answer" in r

def test_adaptive():
    ce = ComplexityEstimator()
    assert ce.top_k("What is RAG?", 5) >= 2
    p = _pipe()
    ar = AdaptiveRetriever(p.retriever)
    res, k = ar.retrieve("Compare supervised and unsupervised learning", base_k=5)
    assert isinstance(res, list)

def test_got():
    p = _pipe()
    got = GraphOfThoughts(p)
    dq = RuleBasedDecomposer().decompose("Compare supervised and unsupervised learning")
    thoughts = got.generate_thoughts("Compare supervised and unsupervised learning", dq)
    assert len(thoughts) > 0
    assert got.aggregate()