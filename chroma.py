import logging
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from time import sleep

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



def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        # openai_api_key = os.environ["OPENAI_API_KEY"]
        # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    proxy_embeddings = EmbeddingProxy(embeddings)
    db = Chroma(collection_name=dental_forum,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", collection_name))
    db.add_documents(texts)

    return db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs