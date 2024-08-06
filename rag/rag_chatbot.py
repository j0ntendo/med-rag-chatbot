import openai
from tools.chromadb_tool import query_chromadb
from tools.duckduckgo_tool import query_duckduckgo

# Make sure to set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def rag_chatbot(user_input, tools):
    """
    Main function to handle Reason-Action-Goal chatbot logic.
    """
    observations = []
    
    # Query ChromaDB if it's in the tools list
    if "ChromaDB" in tools:
        try:
            db_result = query_chromadb(user_input)
            if db_result:
                observations.append(f"ChromaDB Result: {db_result}")
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            observations.append("ChromaDB encountered an error or no result.")
    
    # Query DuckDuckGo if it's in the tools list
    if "DuckDuckGo" in tools:
        try:
            ddg_result = query_duckduckgo(user_input)
            if ddg_result:
                observations.append(f"DuckDuckGo Result: {ddg_result}")
        except Exception as e:
            print(f"Error querying DuckDuckGo: {e}")
            observations.append("DuckDuckGo encountered an error or no result.")
    
    # Construct the prompt
    prompt = construct_rag_prompt(user_input, tools, observations)
    
    # Get GPT response
    response = get_gpt_response(prompt)
    return response

def get_gpt_response(prompt, model="text-davinci-003", max_tokens=150):
    """
    Function to get a response from GPT-3.5 or GPT-4 with a given prompt.
    """
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,  # Adjust if needed
            top_p=1.0,        # Adjust if needed
            n=1               # Number of responses
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return "Sorry, there was an error processing your request."

def construct_rag_prompt(user_input, tools, observations=None):
    """
    Construct a prompt for the RAG model, incorporating tool results.
    """
    tool_names = ", ".join(tools)
    observations_text = "\n".join(observations) if observations else "No observations found."

    prompt = f"""
    Answer the following dental-related question as best you can. You have access to the following tools:

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
