from rag.rag_chatbot import rag_chatbot

def main():
    user_input = "Can I drink cola after getting my wisdom teeth out?"
    tools = ["ChromaDB", "DuckDuckGo"]

    response = rag_chatbot(user_input, tools)
    print("Chatbot Response:")
    print(response)

if __name__ == "__main__":
    main()
