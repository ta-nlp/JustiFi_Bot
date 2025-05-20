#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from typing import Any, List
from langchain.schema import Document
# from langchain.retrievers import BaseRetriever


# In[2]:


def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    BM25 (Best Matching 25) is a ranking function used in information retrieval.
    It's based on the probabilistic retrieval framework and is an improvement over TF-IDF.

    Args:
    documents (List[Document]): List of documents to index.

    Returns:
    BM25Okapi: An index that can be used for BM25 scoring.
    """
    # Tokenize each document by splitting on whitespace
    # This is a simple approach and could be improved with more sophisticated tokenization
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


# In[ ]:


# bm25 = create_bm25_index(cleaned_texts) 


# In[3]:


def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
    vectorstore (VectorStore): The vectorstore containing the documents.
    bm25 (BM25Okapi): Pre-computed BM25 index.
    query (str): The query string.
    k (int): The number of documents to retrieve.
    alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
    List[Document]: The top k documents based on the combined scores.
    """
    
    epsilon = 1e-8

    # Step 1: Get all documents from the vectorstore
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # Step 2: Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Step 3: Perform vector search
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))
    
    # Step 4: Normalize scores
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) -  np.min(bm25_scores) + epsilon)

    # Step 5: Combine scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores  

    # Step 6: Rank documents
    sorted_indices = np.argsort(combined_scores)[::-1]
    
    # Step 7: Return top k documents
    return [all_docs[i] for i in sorted_indices[:k]]


# RERANKING

# In[4]:


class CrossEncoderRetriever(BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    bm25: Any = Field(description="bm25 index")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")
    # query: str = Field(description="Query to be answered")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)

        # retrieval using vector vector_data + bm25 
        initial_docs = fusion_retrieval(self.vectorstore, self.bm25, query, k=self.k, alpha=0.5)
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by score
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top reranked documents
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")


