from together import Together

client = Together(api_key = '19d02f4dba59e2f0a1f8fc20f01cb28b783529206dd847ed5c0c5b83b14929f5')


OPENAI_API_KEY = "your_api_key"

def classify_query(query, conversation_history):
    system_prompt = system_prompt = """You are a classifier that determines if a user query is legal-related or general.

- If the query involves legal terms, court cases, contracts, laws, legal rulings, or any legal-related matters, classify it as 'legal'.
- If the query involves a request for a summary, rephrasing, or any analysis (e.g., "make it in 50 words", "summarize this in 200 words", "rephrase this document"), classify it as 'legal'.
- If the query involves any document analysis or review (e.g., "analyze this document", "review this contract", "identify legal issues in this text"), classify it as 'legal'.
- If the query is asking for general knowledge, casual conversation, or non-legal instructions (e.g., "tell me a joke", "what is the weather today?", "how are you?", "thank you", "welcome"), classify it as 'general'.

Only return 'legal' or 'general'.
"""

    messages = []  # Copy the previous conversation history

    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        # messages=[
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": query}
        # ]
        messages=messages
    )

    classification = response.choices[0].message.content
    return classification == "legal"  # Returns True for legal, False for general

# # Example Usage
# query1 = "What is the penalty for breach of contract?"
# query2 = "you're a good bot."

# print(f"Query: {query1} -> Legal Query: {classify_query_with_llm(query1)}")  # Should return True
# print(f"Query: {query2} -> Legal Query: {classify_query_with_llm(query2)}")  # Should return False
