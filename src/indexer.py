"""
Production RAG Retrieval Pipeline for AmazonHelp Historical Customer Resolutions.
Combines:
1. Dense Semantic Vector Search (FAISS + BAAI/bge-small-en-v1.5 embeddings via FastEmbed)
2. Lexical Keyword Matching (TF-IDF Sparse Matrix Cosine Similarity)
3. Intent-Aware Dynamic Reranking and Policy URL Grounding
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import clean_tweet_text, INTENT_DEFINITIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical deep links for Amazon customer support routing
CANONICAL_ACTION_LINKS = {
    "ORDER_TRACKING_AND_STATUS": "https://amzn.to/track",
    "LATE_OR_MISSING_DELIVERY": "https://amzn.to/orders",
    "REFUND_AND_RETURN_STATUS": "https://amzn.to/returns",
    "DAMAGED_OR_WRONG_ITEM": "https://amzn.to/returns",
    "ACCOUNT_AND_SECURITY": "https://amzn.to/help",
    "PRIME_AND_SUBSCRIPTION": "https://amzn.to/prime-manage",
    "PAYMENT_AND_PROMOTIONS": "https://amzn.to/help",
    "GENERAL_FEEDBACK_OR_CHITCHAT": "https://amzn.to/help",
}


class ResolutionIndexer:
    """
    Hybrid RAG Retriever combining dense FAISS semantic vector search with
    lexical TF-IDF keyword matching over historical AmazonHelp resolution pairs.
    """

    def __init__(
        self,
        kb_path: str = "data/historical_resolutions.json",
        index_dir: str = "data/vector_store",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        dense_weight: float = 0.70,
        lexical_weight: float = 0.30,
    ):
        kb_p = Path(kb_path)
        self.kb_path = kb_p if kb_p.is_absolute() else (PROJECT_ROOT / kb_p)
        
        idx_p = Path(index_dir)
        self.index_dir = idx_p if idx_p.is_absolute() else (PROJECT_ROOT / idx_p)
        
        self.embedding_model_name = embedding_model
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

        # Neural dense embedder (FastEmbed with ONNX runtime)
        self.embedder = TextEmbedding(model_name=embedding_model)
        
        # Lexical vectorizer for keyword match precision
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = None

        self.index: Optional[faiss.Index] = None
        self.items: List[Dict[str, Any]] = []
        self._load_or_build()

    @property
    def _index_path(self) -> Path:
        return self.index_dir / "resolutions.faiss"

    @property
    def _metadata_path(self) -> Path:
        return self.index_dir / "resolutions.metadata.json"

    @property
    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self.kb_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(self.embedding_model_name.encode("utf-8"))
        digest.update(b"hybrid-resolution-chunk-v2")
        return digest.hexdigest()

    def _load_or_build(self) -> None:
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}. Run scripts/build_dataset.py first.")
        
        fingerprint = self._fingerprint()
        if self._index_path.exists() and self._metadata_path.exists() and self._manifest_path.exists():
            try:
                manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                if manifest.get("fingerprint") == fingerprint:
                    self.index = faiss.read_index(str(self._index_path))
                    self.items = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                    if self.index.ntotal == len(self.items):
                        # Fit lexical TF-IDF on retrieval texts
                        retrieval_texts = [item.get("retrieval_text", item.get("customer_query", "")) for item in self.items]
                        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(retrieval_texts)
                        return
            except (OSError, ValueError, RuntimeError):
                pass

        self._build(fingerprint)

    def _chunk(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        query = clean_tweet_text(item.get("customer_query", ""))
        reply = clean_tweet_text(item.get("support_reply", ""))
        history = [clean_tweet_text(turn) for turn in item.get("history", [])]
        history = [turn for turn in history if turn]
        if not query or not reply:
            return None
        
        intent = item.get("intent", "GENERAL_FEEDBACK_OR_CHITCHAT")
        canonical_link = CANONICAL_ACTION_LINKS.get(intent, "https://amzn.to/help")
        retrieval_text = f"{history[-1]} {query}" if history else query

        return {
            "id": item.get("id", ""),
            "intent": intent,
            "customer_query": query,
            "support_reply": reply,
            "history": history,
            "canonical_link": canonical_link,
            "retrieval_text": retrieval_text,
        }

    def _embed(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Computes normalized dense embeddings with FastEmbed on CPU in-process.
        
        NOTE: parallel=None is required on Windows to prevent multiprocessing
        worker deadlocks at module import time. Do not change to parallel>0.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            emb_chunk = list(self.embedder.embed(chunk, batch_size=batch_size, parallel=None))
            all_embeddings.extend(emb_chunk)
            print(f"[RAG Indexer] Embedded {min(i + batch_size, len(texts))}/{len(texts)} items...", flush=True)

        vectors = np.asarray(all_embeddings, dtype="float32")
        if vectors.ndim != 2 or not len(vectors):
            raise RuntimeError("Embedding model returned no usable vectors.")
        faiss.normalize_L2(vectors)
        return vectors

    def _build(self, fingerprint: str) -> None:
        raw_items = json.loads(self.kb_path.read_text(encoding="utf-8"))
        self.items = [chunk for item in raw_items if (chunk := self._chunk(item))]
        if not self.items:
            raise ValueError("No valid customer-resolution chunks found in the knowledge base.")

        retrieval_texts = [item["retrieval_text"] for item in self.items]
        print(f"[RAG Indexer] Building Hybrid FAISS + Lexical index for {len(retrieval_texts)} historical resolutions...", flush=True)
        
        # Build dense index
        vectors = self._embed(retrieval_texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # Normalized vectors => Cosine Similarity
        self.index.add(vectors)

        # Build sparse lexical index
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(retrieval_texts)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))
        self._metadata_path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")
        self._manifest_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "embedding_model": self.embedding_model_name,
            "retrieval_strategy": "Hybrid (Dense FAISS Cosine + Lexical TF-IDF)",
            "document_count": len(self.items),
        }, indent=2), encoding="utf-8")
        print(f"[RAG Indexer] Successfully persisted vector index to {self.index_dir}/", flush=True)

    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        intent_filter: Optional[str] = None,
        history: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Semantic Retrieval:
        Combines dense vector similarity with lexical keyword overlap,
        boosts intent-aligned matches, and returns sorted resolutions with full provenance.
        """
        cleaned_query = clean_tweet_text(query)
        if not cleaned_query or not self.index or self.index.ntotal == 0 or top_k < 1:
            return []

        # Formulate full search context including multi-turn history
        search_context = cleaned_query
        if history:
            clean_history = [clean_tweet_text(h) for h in history[-3:] if clean_tweet_text(h)]
            if clean_history:
                search_context = " ".join(clean_history + [cleaned_query])

        # 1. Dense Semantic Search (Cosine similarity)
        candidate_count = min(self.index.ntotal, max(top_k * 10, 40))
        query_vec = self._embed([search_context])
        dense_scores, dense_indices = self.index.search(query_vec, candidate_count)

        # 2. Lexical TF-IDF Search (Sparse Cosine similarity)
        lexical_vec = self.tfidf_vectorizer.transform([search_context])
        lexical_sims = cosine_similarity(lexical_vec, self.tfidf_matrix).flatten()

        # 3. Hybrid Score Fusion & Intent Reranking
        scored_candidates = []
        seen_indices = set()

        for d_score, idx in zip(dense_scores[0], dense_indices[0]):
            if idx < 0:
                continue
            idx = int(idx)
            seen_indices.add(idx)
            
            dense_sim = float(max(0.0, d_score))
            lexical_sim = float(lexical_sims[idx]) if self.tfidf_matrix is not None else 0.0
            
            # Hybrid combined score
            hybrid_score = (self.dense_weight * dense_sim) + (self.lexical_weight * lexical_sim)
            
            item = self.items[idx].copy()
            item_intent = item.get("intent")
            
            # Intent alignment bonus (ensures relevant domain policies are prioritized)
            is_intent_match = (intent_filter is not None and item_intent == intent_filter)
            if is_intent_match:
                hybrid_score += 0.15

            item["dense_similarity"] = round(dense_sim, 4)
            item["lexical_similarity"] = round(lexical_sim, 4)
            item["similarity_score"] = round(hybrid_score, 4)
            item["is_intent_match"] = is_intent_match
            item["retrieval_metric"] = "hybrid_dense_lexical"
            scored_candidates.append(item)

        # Sort candidates by combined hybrid score
        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Separate intent matches and fallbacks to ensure intent fidelity
        intent_matches = [c for c in scored_candidates if not intent_filter or c["intent"] == intent_filter]
        fallbacks = [c for c in scored_candidates if intent_filter and c["intent"] != intent_filter]

        results = (intent_matches + fallbacks)[:top_k]
        return results


# Export alias
RAGPipeline = ResolutionIndexer
