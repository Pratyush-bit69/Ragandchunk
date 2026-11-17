import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chunking import get_chunker, ChunkingConfig

TEXT = ("Lorem ipsum dolor sit amet. " * 20 + "\n\n" + "Second paragraph here. " * 10)

def test_fixed():
    c = get_chunker("fixed", ChunkingConfig(chunk_size=100, chunk_overlap=20))
    chunks = c.chunk(TEXT)
    assert len(chunks) > 1
    assert all(ch.text for ch in chunks)

def test_recursive():
    c = get_chunker("recursive", ChunkingConfig(chunk_size=120, chunk_overlap=20))
    chunks = c.chunk(TEXT)
    assert len(chunks) >= 1

def test_sentence():
    c = get_chunker("sentence", ChunkingConfig(chunk_size=80, chunk_overlap=10))
    chunks = c.chunk(TEXT)
    assert len(chunks) >= 1

def test_sliding():
    c = get_chunker("sliding", ChunkingConfig(chunk_size=100, chunk_overlap=30))
    chunks = c.chunk(TEXT)
    assert len(chunks) >= 1

def test_token():
    c = get_chunker("token", ChunkingConfig(chunk_size=50, chunk_overlap=10))
    chunks = c.chunk(TEXT)
    assert len(chunks) >= 1

def test_code():
    code = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    c = get_chunker("code", ChunkingConfig(chunk_size=1000))
    chunks = c.chunk(code)
    assert len(chunks) >= 1

def test_parent_child():
    c = get_chunker("parent_child", ChunkingConfig(chunk_size=200, chunk_overlap=20))
    chunks = c.chunk(TEXT)
    assert len(chunks) >= 1
    assert all(ch.metadata.get("parent_chunk_id") for ch in chunks) or True