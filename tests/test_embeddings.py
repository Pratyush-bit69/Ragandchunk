import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rag.embeddings import HashingEmbeddings, TFIDFEmbeddings

def test_hashing():
    e = HashingEmbeddings(dim=64)
    v = e.embed(["hello world", "foo bar"])
    assert v.shape == (2, 64)

def test_tfidf():
    e = TFIDFEmbeddings(max_features=100)
    corpus = ["hello world", "foo bar baz", "hello foo"]
    v = e.embed(corpus)
    assert v.shape[0] == 3