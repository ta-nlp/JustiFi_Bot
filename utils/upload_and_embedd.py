#!/usr/bin/env python
# coding: utf-8

# In[8]:


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.schema.embeddings import Embeddings


# In[2]:


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

# Create a custom embedding class
class InLegalBERTEmbeddings(Embeddings):
    def embed_documents(self, texts):
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded_input)
            embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
        return embeddings
    
    def embed_query(self, query):
        return self.embed_documents([query])[0]

# Instantiate your custom embeddings
embedding_model = InLegalBERTEmbeddings()


# In[3]:


# Function to extract text from uploaded files
def extract_text_from_file(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".doc"):
        loader = UnstructuredFileLoader(file_path)
    else:
        print("Unsupported file format")
    
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


# In[4]:


# Function to process local files with os library
def upload_files(files: list[str], chunk_size, chunk_overlap):
    all_text = ""
    documents = []
    for file_path in files:
        # Extract text from the file
        # text = extract_text_from_file(file_path)  # Assuming extract_text_from_file is defined
        # all_text += text + "\n"
        doc = extract_text_from_file(file_path)
        documents.append(Document(page_content=doc))
    
    # Split text into chunks for embedding (using your splitter)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # documents = text_splitter.create_documents([all_text])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    # Initialize an empty list to store all the chunks
    all_texts = []
    
    # Iterate over the list of documents and split each one
    for document in documents:
        chunks = text_splitter.split_documents([document])  # Split each document
        all_texts.extend(chunks)  # Add the resulting chunks to the all_texts list
         # Get the last chunk (if there are any chunks)
        if chunks:
            last_chunk = chunks[-1]  # Get the last chunk
            
            # Modify the last chunk (example: change its content)
            modified_last_chunk = last_chunk.page_content + "The important information about the date, place and concerned personals (such as judges in the bench, date and place of judgement) may be found here."  # Example modification
            
            # Update the last chunk with the modified content
            chunks[-1] = Document(page_content=modified_last_chunk)
            
            # If you need the modified chunk to be added back to the all_texts list, you can do that too
            all_texts[-1] = chunks[-1]  # Update the last item in all_texts with the modified chunk

    vector_store = FAISS.from_documents(all_texts, embedding_model)
    
    # Store in FAISS vector DB
    # if vector_store is not None:
    #     vector_store.add_documents(documents)
    # else:
    # vector_store = FAISS.from_documents(documents, embedding_model)
    # print(vector_store)
    print("Files uploaded and processed successfully.")
    return all_texts, vector_store


