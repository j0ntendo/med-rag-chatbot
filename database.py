
# "mixedbread-ai/mxbai-embed-large-v1"

import logging
import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from time import sleep
import json
import hashlib
from uuid import uuid4
from langchain_core.documents import Document

EMBED_DELAY = 0.02  # 20 milliseconds

class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)

def create_vector_db(collection_name="chroma"):
    embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    proxy_embeddings = EmbeddingProxy(embeddings)
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=proxy_embeddings,
        persist_directory=os.path.join("DATA/", collection_name)
    )
    return vector_store

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def generate_hash(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def process_documents(json_file_path, vector_store):
    json_data = load_json_file(json_file_path)
    
    documents_to_add = []
    uuids = []

    for entry in json_data:
        doc_id = str(entry['id'])
        content_hash = generate_hash(entry['dialogue'])
        
        doc = Document(
            page_content=entry['dialogue'],
            metadata={"title": entry['title'], "content_hash": content_hash},
            id=doc_id
        )
        documents_to_add.append(doc)
        uuids.append(doc_id)
    
    if documents_to_add:
        vector_store.add_documents(documents=documents_to_add, ids=uuids)

# Example usage
if __name__ == "__main__":
    load_dotenv()  # Load environment variables
    vector_store = create_vector_db(collection_name="dental_forum")
    process_documents('DATA/dental_QA.json', vector_store)
