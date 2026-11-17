# RAG, Chunking, Evaluation, and Graph

A self-contained Python project implementing a full Retrieval-Augmented Generation (RAG) system with:

- **Document loading** (txt, md, csv, json, html, directories)
- **Chunking strategies** (fixed, recursive, sentence, paragraph, sliding window, token, semantic, parent-child, code)
- **Embeddings** (sentence-transformers, TF-IDF, hashing, OpenAI)
- **Vector stores** (in-memory, Chroma)
- **Retrievers** (dense, sparse/BM25, hybrid, rerank, MMR, multi-query, parent-document)
- **Knowledge graph + GraphRAG** (entity/relation extraction, traversal, community detection, visualization)
- **Generation** (extractive, OpenAI, Anthropic)
- **Evaluation** (retrieval metrics: P@k, R@k, MRR, nDCG, MAP; generation metrics: faithfulness, answer relevance, context precision/recall, token F1, ROUGE-L)
- **Interfaces** (CLI scripts, FastAPI, Streamlit)

## Setup

```bash
cd /projects/Ragandchunk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick demo

```bash
.venv/bin/python scripts/demo.py
```

Produces `outputs/` with an eval report and knowledge graph image.

## Compare chunking strategies

```bash
.venv/bin/python scripts/compare_chunking.py
```

## Evaluate

```bash
.venv/bin/python scripts/evaluate.py
```

## API

```bash
.venv/bin/python api.py
# POST /init {"data_dir": "data/sample_docs"}
# POST /query {"query": "What is RAG?"}
```

## Streamlit UI

```bash
.venv/bin/streamlit run app.py
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Configuration

All components are configurable via `rag/config.py` (`RAGConfig`) or environment variables:
`RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL`, `RAG_GENERATION_PROVIDER`, `RAG_GENERATION_MODEL`, `RAG_PERSIST_DIR`.

## Advanced capabilities (`rag/advanced/`)

Innovative modules targeting complex, unsolved RAG problems:

- `agentic_rag.py` — Agentic RAG: plan → retrieve → grade → rewrite → synthesize → self-critique, with abstention
- `query_decomposition.py` — multi-hop question decomposition
- `hyde.py` — Hypothetical Document Embeddings
- `retrieval_grader.py` — LLM/heuristic retrieval grading
- `query_rewriter.py` — query rewriting for better recall
- `conflict.py` — conflicting-source detection + provenance attribution
- `confidence.py` — confidence scoring + calibrated abstention
- `late_interaction.py` — ColBERT-style MaxSim retriever
- `temporal_graph.py` — temporal knowledge graph
- `adaptive_chunking.py` — structure/density-aware adaptive chunking
- `verified_generation.py` — NLI-style faithfulness gating
- `multi_agent_debate.py` — multi-agent debate RAG
- `self_consistency.py` — self-consistency sampling & voting
- `adaptive_retrieval.py` — complexity-based adaptive retrieval depth
- `graph_of_thoughts.py` — graph-of-thoughts reasoning
- `complex_evaluation.py` — multi-hop, conflict, abstention, provenance metrics
- `llm_judge.py` — LLM-as-a-judge evaluation (faithfulness, relevance, correctness)
- `llm_reranker.py` — LLM-based reranking
- `conversational_rag.py` — multi-turn RAG with query condensation & memory
- `context_compression.py` — extractive/LLM context compression
- `query_expansion.py` — LLM query expansion for improved recall

Run: `.venv/bin/python scripts/advanced_demo.py`

## Project layout

```
rag/            core library (incl. advanced/)
data/           sample documents + eval dataset
scripts/        CLI entry points
tests/          pytest tests
app.py          Streamlit UI
api.py          FastAPI service
```