"""Document loaders for various formats."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            import hashlib
            self.doc_id = hashlib.md5(self.content.encode("utf-8")).hexdigest()[:12]


class BaseLoader:
    def load(self, source) -> List[Document]:
        raise NotImplementedError


class TextLoader(BaseLoader):
    def load(self, path) -> List[Document]:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        return [Document(content=text, metadata={"source": str(p), "type": "text"})]


class MarkdownLoader(TextLoader):
    def load(self, path) -> List[Document]:
        docs = super().load(path)
        for d in docs:
            d.metadata["type"] = "markdown"
        return docs


class CSVLoader(BaseLoader):
    def __init__(self, content_column: Optional[str] = None, metadata_columns: Optional[List[str]] = None):
        self.content_column = content_column
        self.metadata_columns = metadata_columns or []

    def load(self, path) -> List[Document]:
        p = Path(path)
        docs = []
        with p.open(newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if self.content_column and self.content_column in row:
                    content = row[self.content_column]
                else:
                    content = " | ".join(f"{k}: {v}" for k, v in row.items())
                meta = {"source": str(p), "row": i, "type": "csv"}
                for c in self.metadata_columns:
                    if c in row:
                        meta[c] = row[c]
                docs.append(Document(content=content, metadata=meta))
        return docs


class JSONLoader(BaseLoader):
    def __init__(self, content_key: str = "content", metadata_keys: Optional[List[str]] = None):
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []

    def load(self, path) -> List[Document]:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        docs = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                content = item.get(self.content_key, json.dumps(item))
                meta = {"source": str(p), "index": i, "type": "json"}
                for k in self.metadata_keys:
                    if k in item:
                        meta[k] = item[k]
            else:
                content = str(item)
                meta = {"source": str(p), "index": i, "type": "json"}
            docs.append(Document(content=content, metadata=meta))
        return docs


class HTMLLoader(BaseLoader):
    def load(self, path) -> List[Document]:
        from bs4 import BeautifulSoup
        p = Path(path)
        html = p.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        content = "\n".join(lines)
        return [Document(content=content, metadata={"source": str(p), "type": "html", "title": soup.title.string if soup.title else None})]


class DirectoryLoader(BaseLoader):
    EXT_MAP = {
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
    }

    def __init__(self, glob_pattern: str = "**/*", recursive: bool = True):
        self.glob_pattern = glob_pattern
        self.recursive = recursive

    def load(self, path) -> List[Document]:
        p = Path(path)
        docs = []
        pattern = self.glob_pattern
        for f in p.glob(pattern):
            if not f.is_file():
                continue
            loader_cls = self.EXT_MAP.get(f.suffix.lower())
            if loader_cls is None:
                continue
            try:
                docs.extend(loader_cls().load(f))
            except Exception as e:
                print(f"[DirectoryLoader] skip {f}: {e}")
        return docs


def load_document(path, **kwargs) -> List[Document]:
    p = Path(path)
    if p.is_dir():
        return DirectoryLoader(**kwargs).load(p)
    suffix = p.suffix.lower()
    mapping = {
        ".txt": TextLoader, ".md": MarkdownLoader, ".markdown": MarkdownLoader,
        ".csv": CSVLoader, ".json": JSONLoader, ".html": HTMLLoader, ".htm": HTMLLoader,
    }
    cls = mapping.get(suffix, TextLoader)
    return cls(**kwargs).load(p)