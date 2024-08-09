import os
import sys
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from tools.chromadata import retrieve_bm25
from tools.prompt.react_template import get_react_template

# set up for openai
os.environ['OPENAI_API_KEY'] = 
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

tools = [retrieve_bm25]
prompt = get_react_template()

react_agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
)

user_input = "can i drink cola after wisdom teeth removal?"
agent_executor.invoke({"input": user_input})





# # example of main.py
# if __name__ == "__main__":

#     result = rag_chatbot(user_input)
#     print(result)
