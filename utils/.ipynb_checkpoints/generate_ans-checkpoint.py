#!/usr/bin/env python
# coding: utf-8

# In[1]:


from together import Together
from rank_bm25 import BM25Okapi
from utils.retrieve_and_rerank import CrossEncoderRetriever
from sentence_transformers import CrossEncoder


# In[2]:


vector_store = None

# TogetherAI API setup (Replace with API key)
client = Together(api_key = '19d02f4dba59e2f0a1f8fc20f01cb28b783529206dd847ed5c0c5b83b14929f5')
# client = Client( host='http://127.0.0.1:11434')


# In[ ]:


cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# In[ ]:


# Endpoint to handle user queries
def query_model(bm25: BM25Okapi, vector_store, question: str = None):
    # Retrieve relevant chunks
    # docs = vector_store.similarity_search(question, k=5)
    # context = "\n".join([doc.page_content for doc in docs])

    # Create the cross-encoder retriever
    cross_encoder_retriever = CrossEncoderRetriever(
        vectorstore= vector_store,
        cross_encoder=cross_encoder,
        bm25=bm25,
        k=10,  # Retrieve 10 documents initially
        rerank_top_k=10  # Return top 5 after reranking
    )

    reranked_docs = cross_encoder_retriever.get_relevant_documents(question)
    reranked_context = "\n".join([doc.page_content for doc in reranked_docs])  # Joining the content of the reranked docs

    # print(reranked_context)

    response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    # model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    # model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    messages = [
        {"role": "system", "content": "You are a highly skilled chatbot focused on understanding and interpreting legal judgments. Your task is to accurately analyze legal documents, extract relevant facts, and provide precise and clear legal answers based on the context provided. You should be able to recognize and handle legal synonyms or equivalent legal terms, ensuring that different expressions of the same concept are interpreted correctly."},
        {"role": "user", "content": f"Answer the following legal query based on the provided judgment document: {question}. Use only the information from the document below, and ensure you account for any legal synonyms or equivalent terms that may appear in the query. Provide an accurate response based solely on the document: {reranked_context}"}
    ],
    temperature=0,
    top_p=0,
    top_k=1
    )
    
    # print(response)
    answer = response.choices[0].message.content if response.choices else "No response."
    return answer
 

