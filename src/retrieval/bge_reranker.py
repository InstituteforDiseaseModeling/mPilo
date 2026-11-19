from __future__ import annotations
from typing import Optional, Sequence
from langchain.schema import Document
from langchain.pydantic_v1 import Extra

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from sentence_transformers import CrossEncoder


class BgeReranker(BaseDocumentCompressor):
    """LangChain implementation of BAII/bge-reranker-base (https://huggingface.co/BAAI/bge-reranker-base)."""

    model_name: str = "BAAI/bge-reranker-base"
    """Model name to use for reranking."""
    top_n: int = 10
    """Number of documents to return."""
    model: CrossEncoder = CrossEncoder(model_name)
    """CrossEncoder instance to use for reranking."""

    def bge_rerank(self, query, docs):
        """Run the reranker model."""
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using BAAI/bge-reranker models."""
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results


class BgeRerankerLarge(BaseDocumentCompressor):
    """LangChain implementation of BAII/bge-reranker-large (https://huggingface.co/BAAI/bge-reranker-large)."""

    model_name: str = "BAAI/bge-reranker-large"
    """Model name to use for reranking."""
    top_n: int = 10
    """Number of documents to return."""
    model: CrossEncoder = CrossEncoder(model_name)
    """CrossEncoder instance to use for reranking."""

    def bge_rerank(self, query, docs):
        """Run the reranker model."""
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models."""
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results
