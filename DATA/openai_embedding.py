import logging
import os
import json
from uuid import uuid4
from time import sleep
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


logging.basicConfig(level=logging.INFO)


os.environ["OPENAI_API_KEY"] = 


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = Chroma(
    collection_name="dental_forum",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_documents(data):
    """Create a list of Document objects from JSON data with UUIDs."""
    documents = []
    uuids = []
    for item in data:
        doc_id = str(uuid4())  
        doc = Document(
            page_content=item['dialogue'],  
            metadata={"title": item.get('title', '')},
            id=doc_id  
        )
        documents.append(doc)
        uuids.append(doc_id)
    return documents, uuids

def add_documents_to_chroma(documents, uuids):
    """Add documents to ChromaDB with embeddings."""
    vector_store.add_documents(documents=documents, ids=uuids)
    logging.info("Documents have been added to ChromaDB.")

def main():
    
    json_data = load_json('DATA/dental_QA.json')
    
    
    documents, uuids = create_documents(json_data)
    
    
    add_documents_to_chroma(documents, uuids)
    
    logging.info("All documents have been added to ChromaDB.")

if __name__ == "__main__":
    main()
