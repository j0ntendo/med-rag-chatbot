import getpass
import os
import json
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain import hub
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import hashlib
import streamlit as st

os.environ['OPENAI_API_KEY'] = 

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def create_bm25_retriever(vector_store):
    docs = [Document(page_content=doc.page_content, id=doc.id) for doc in vector_store.get_all_documents()]
    return BM25Retriever.from_documents(docs)

def construct_rag_prompt(user_input, tools, observations=None):
    tool_names = ", ".join(tools)
    observations_text = "\n".join(observations) if observations else "No observations found."

    prompt = f"""
    Answer the following question as best you can. You have access to the following tools:

    {tool_names}

    Use the following format:

    Question: {user_input}
    Thought: Always think about what to do next.
    Action: The action to take, should be one of [{tool_names}]
    Action Input: The input to the action
    Observation: The result of the action
    {observations_text}
    Thought: I now know the final answer.
    Final Answer: The final answer to the original input question.

    Begin!

    Question: {user_input}
    Thought:
    """
    return prompt

def query_chromadb(query, vector_store):
    retriever = create_bm25_retriever(vector_store)
    result = retriever.invoke(query)
    return result

def query_duckduckgo(query):
    return "DuckDuckGo result placeholder"

def setup_rag_chain(llm, tools):
    prompt = hub.pull("rlm/rag-prompt")  
    react_agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    return agent_executor

def rag_chatbot(user_input, tools):
    observations = []
    
    vector_store = create_vector_db(collection_name="dental_forum")
    
    if "ChromaDB" in tools:
        try:
            db_result = query_chromadb(user_input, vector_store)
            if db_result:
                observations.append(f"ChromaDB Result: {db_result}")
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            observations.append("ChromaDB encountered an error or no result.")
    
    if "DuckDuckGo" in tools:
        try:
            ddg_result = query_duckduckgo(user_input)
            if ddg_result:
                observations.append(f"DuckDuckGo Result: {ddg_result}")
        except Exception as e:
            print(f"Error querying DuckDuckGo: {e}")
            observations.append("DuckDuckGo encountered an error or no result.")
    
    agent_executor = setup_rag_chain(llm, tools)
    
    response = agent_executor.run({"question": user_input, "observations": observations})
    return response

if __name__ == "__main__":
    user_input = "can i drink cola after wisdom teeth removal?"
    tools = ["ChromaDB"]
    result = rag_chatbot(user_input, tools)
    print(result)
