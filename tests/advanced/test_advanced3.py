import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig, GenerationConfig
from rag.advanced.llm_judge import LLMJudge
from rag.advanced.llm_reranker import LLMReranker
from rag.advanced.conversational_rag import ConversationalRAG
from rag.advanced.context_compression import ExtractiveCompressor, LLMCompressor
from rag.advanced.query_expansion import HybridQueryExpander

def _pipe(gen_provider="extractive"):
    cfg = RAGConfig(); cfg.chunking = ChunkingConfig(strategy="recursive", chunk_size=200, chunk_overlap=30)
    cfg.retrieval = RetrievalConfig(mode="hybrid", top_k=3)
    cfg.generation = GenerationConfig(provider=gen_provider, model="Qwen/Qwen2.5-0.5B-Instruct")
    p = RAGPipeline(cfg); p.ingest_text("Machine learning includes supervised and unsupervised learning. RAG combines retrieval and generation. Evaluation uses precision and recall.", metadata={"source":"t"})
    return p

def test_judge_extractive():
    p = _pipe()
    from rag.generation import ExtractiveGenerator
    g = ExtractiveGenerator()
    judge = LLMJudge(g)
    r = judge.evaluate("What is RAG?", "RAG combines retrieval and generation.", ["RAG combines retrieval and generation."])
    assert "llm_faithfulness" in r

def test_reranker():
    p = _pipe()
    from rag.generation import ExtractiveGenerator
    rr = LLMReranker(ExtractiveGenerator(), top_k=2)
    res = p.retrieve("RAG", top_k=3)
    out = rr.rerank("RAG", res)
    assert len(out) <= 2

def test_conversational():
    p = _pipe()
    conv = ConversationalRAG(p)
    r1 = conv.ask("What is RAG?", top_k=2)
    assert "answer" in r1
    r2 = conv.ask("How is it evaluated?", top_k=2)
    assert "standalone_query" in r2

def test_compression():
    p = _pipe()
    comp = ExtractiveCompressor(max_chars=100)
    ctxs = [c for c,_ in p.retrieve("RAG", top_k=2)]
    out = comp.compress("RAG", ctxs)
    assert len(out) == len(ctxs)

def test_query_expansion():
    from rag.generation import ExtractiveGenerator
    ex = HybridQueryExpander()
    variants = ex.expand("What is RAG?")
    assert len(variants) >= 1