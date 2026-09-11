"""
Production RAG Retrieval Module powered by FastEmbed and FAISS.
Provides drop-in compatibility and helper interfaces for dense and hybrid semantic retrieval
over historical customer support conversations.

Exports:
- FastEmbedFAISSRetriever: Primary retriever class.
- RAGPipeline: Alias for FastEmbedFAISSRetriever (maintains backward compatibility).
- LangChainFAISSRetriever: Alias for FastEmbedFAISSRetriever for LangChain integrations.
"""

from typing import List, Dict, Any, Optional
from src.indexer import ResolutionIndexer, CANONICAL_ACTION_LINKS

class FastEmbedFAISSRetriever:
    """
    RAG retriever wrapper around ResolutionIndexer for semantic resolution search.
    """
    def __init__(
        self,
        kb_path: str = "data/historical_resolutions.json",
        index_dir: str = "data/vector_store"
    ):
        self.indexer = ResolutionIndexer(kb_path=kb_path, index_dir=index_dir)

    def retrieve_similar(
        self,
        query: str,
        top_k: int = 3,
        intent_filter: Optional[str] = None,
        history: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k historical resolutions semantically similar to the query.
        """
        return self.indexer.retrieve(
            query=query,
            top_k=top_k,
            intent_filter=intent_filter,
            history=history
        )


LangChainFAISSRetriever = FastEmbedFAISSRetriever
RAGPipeline = FastEmbedFAISSRetriever
