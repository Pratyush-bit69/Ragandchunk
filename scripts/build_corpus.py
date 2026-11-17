"""Fetch a diverse set of Wikipedia articles to build a larger, real-world corpus."""
from __future__ import annotations

import json, os, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "corpus")
os.makedirs(OUT_DIR, exist_ok=True)

TOPICS = [
    "Photosynthesis", "Machine learning", "Climate change", "World War II",
    "DNA", "Quantum computing", "Black hole", "Renaissance", "Internet",
    "Vaccine", "Blockchain", "CRISPR", "Solar System", "Neural network",
    "Democracy", "Antibiotic", "Plate tectonics", "Relativity", "Ecosystem",
    "Artificial intelligence",
]

def fetch(title):
    url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=" + urllib.parse.quote(title) + "&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "RAGForge/1.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    pages = data["query"]["pages"]
    page = list(pages.values())[0]
    return page.get("extract", "")

def main():
    for t in TOPICS:
        try:
            text = fetch(t)
            if not text or len(text) < 500:
                print("SKIP (too short):", t)
                continue
            # trim to ~6000 chars to keep chunks manageable
            text = text[:6000]
            slug = t.lower().replace(" ", "_")
            path = os.path.join(OUT_DIR, slug + ".md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {t}\n\n")
                f.write(text)
            print(f"OK {t}: {len(text)} chars -> {path}")
            time.sleep(2.5)
        except Exception as e:
            print("ERR", t, e)

if __name__ == "__main__":
    main()