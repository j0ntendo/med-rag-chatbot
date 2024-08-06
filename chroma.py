import chromadb
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from chromadb.utils.embedding_functions import LlamaIndexEmbeddingAdapter
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex
from IPython.display import Markdown, display

mixedbread_api_key = "emb_efdc2ae945a35f9691a12529cefe0896d9d0ef0ed8e96f5c"
model_name = "mixedbread-ai/mxbai-embed-large-v1"
db_path = "./chroma_db"

# Initialize the embedding model
embed_model = FastEmbedEmbedding(model_name=model_name)

def create_and_store_embeddings(data):
    """
    Create embeddings for the data and store them in a persistent ChromaDB collection.

    Args:
        data: The data to process.
    """
    # Initialize the ChromaDB persistent client
    db = chromadb.PersistentClient(path=db_path)
    
    # Get or create a collection with cosine similarity configured
    chroma_collection = db.create_collection(
        name="dental_forum",
        metadata={"hnsw:space": "cosine"}  # Configuring cosine similarity
    )
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    documents = []
    for example in data:
        text_to_embed = example.get('dialogue')
        if text_to_embed:
            documents.append(text_to_embed)
            chroma_collection.add(
                ids=[str(example['id'])],
                documents=[text_to_embed],
                metadatas=[{"title": example.get('title', 'Untitled')}]
            )
    
    # Create and return a VectorStoreIndex
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    
    return index

def load_from_disk():
    """
    Load data from a persistent ChromaDB collection.

    Returns:
        A VectorStoreIndex initialized with the data.
    """
    db2 = chromadb.PersistentClient(path=db_path)
    chroma_collection = db2.get_or_create_collection("dental_forum")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

# if __name__ == "__main__":
#     try:
#         json_data = load_json_data("path/to/dental_QA.json")
#         index = create_and_store_embeddings(json_data)
#         print("Embeddings created and stored successfully.")
#         index = load_from_disk()
#         query_engine = index.as_query_engine()
#         response = query_engine.query("can i drink cola after tooth extraction")
#         print(f"Response: {response}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

