from langchain.agents import tool
import chromadb
from langchain.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


COLLECTION_NAME = "dental_forum"
PERSIST_DIRECTORY = "./chroma_langchain_db"

@tool
def retrieve_bm25(query: str):
    """
    Retrieves documents from the fixed Chroma collection using a BM25 retriever.

    Args:
        query (str): The query string to search for.

    Returns:
        list: A list of documents that match the query.
    """
    
    db = chromadb.PersistentClient(path="./chroma_langchain_db")
    
    documents = db.get_all_documents()

    
    bm25_docs = [Document(page_content=doc.page_content, id=str(doc.id)) for doc in documents]

    
    retriever = BM25Retriever.from_documents(bm25_docs)

    
    results = retriever.invoke(query)
    
    return results
