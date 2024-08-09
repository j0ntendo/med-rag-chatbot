import json
import tiktoken

# Initialize the tokenizer for the model you're using
tokenizer = tiktoken.get_encoding("cl100k_base")  # For "text-embedding-3-large"

def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def calculate_total_tokens(data):
    total_tokens = 0
    for item in data:
        dialogue = item.get('dialogue', '')  # Adjust based on your JSON structure
        token_count = count_tokens(dialogue)
        total_tokens += token_count
        print(f"ID {item['id']} has {token_count} tokens")
    return total_tokens

# Load JSON data from file
with open('DATA/dental_QA.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Calculate the total number of tokens required
total_tokens = calculate_total_tokens(json_data)

print(f"\nTotal tokens required for embedding: {total_tokens}")
