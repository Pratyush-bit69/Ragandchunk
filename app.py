"""Streamlit UI for RAG exploration."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Explorer", layout="wide")
st.title("RAG Explorer")

if "pipe" not in st.session_state:
    st.session_state.pipe = None

with st.sidebar:
    st.header("Configuration")
    data_dir = st.text_input("Data directory", value="data/sample_docs")
    strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "sentence", "semantic", "sliding", "token", "parent_child"])
    chunk_size = st.slider("Chunk size", 100, 1000, 400, 50)
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"])
    top_k = st.slider("Top K", 1, 10, 5)
    if st.button("Build index"):
        with st.spinner("Ingesting..."):
            cfg = RAGConfig()
            cfg.chunking = ChunkingConfig(strategy=strategy, chunk_size=chunk_size)
            cfg.retrieval = RetrievalConfig(mode=mode, top_k=top_k)
            pipe = RAGPipeline(cfg)
            pipe.ingest(data_dir, chunking_strategy=strategy)
            st.session_state.pipe = pipe
        st.success(f"Indexed {len(pipe.chunks)} chunks")

pipe = st.session_state.pipe
if pipe is None:
    st.info("Build an index from the sidebar to start.")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Query")
    query = st.text_input("Question")
    if query:
        res = pipe.query(query, top_k=top_k)
        st.markdown("**Answer**")
        st.write(res.get("answer"))
        st.markdown("**Retrieved chunks**")
        for c, s in res.get("retrieved", []):
            with st.expander(f"{c.chunk_id} (score {s:.3f})"):
                st.write(c.text)
with col2:
    st.subheader("Graph stats")
    st.json(pipe.graph_stats())
    st.subheader("Chunks")
    st.write(len(pipe.chunks))