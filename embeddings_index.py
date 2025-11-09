# embeddings_index.py
import os
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

EMBED_BATCH_SIZE = 64

class EmbeddingsIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.passages: List[str] = []
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def build(self, passages: List[str]):
        if not passages:
            raise ValueError("No passages to index.")
        self.passages = passages
        vectors = self.embedder.encode(passages, show_progress_bar=True, convert_to_numpy=True, batch_size=EMBED_BATCH_SIZE)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    def save(self, index_path: str, passages_path: str):
        if self.index is None:
            raise RuntimeError("Index not built.")
        faiss.write_index(self.index, index_path)
        with open(passages_path, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, passages_path: str):
        if not os.path.exists(index_path) or not os.path.exists(passages_path):
            raise FileNotFoundError("Index or passages file not found.")
        self.index = faiss.read_index(index_path)
        with open(passages_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)

    def query(self, q: str, top_k: int = 4):
        if self.index is None:
            raise RuntimeError("Index not built/loaded.")
        q_vec = self.embedder.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, top_k)
        results = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0 or idx >= len(self.passages):
                continue
            results.append({"id": idx, "score": float(score), "passage": self.passages[idx]})
        return results
