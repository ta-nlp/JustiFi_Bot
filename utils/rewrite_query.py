from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_together import ChatTogether

# Use Together.ai LLM
llm_together = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # or llama-3, gemma, etc.
    temperature=0.4,
    together_api_key="19d02f4dba59e2f0a1f8fc20f01cb28b783529206dd847ed5c0c5b83b14929f5"
)

# Updated template to include chat history
query_rewrite_template = """
        You are an AI assistant tasked with reformulating user queries to improve classification and retrieval in a RAG-based system.

        Your goal is to rewrite the user’s current query to be more specific, unambiguous, and informative by using prior chat history and inferred intent. 
        Do not explain your reasoning—only output the rewritten query directly.
        
        - If chat history is empty and the original query lacks a clear topic, return the original query unchanged.
        - Assume that the user query relates to the prior conversation unless it clearly doesn't.
        - If the user is asking for a summary or explanation, infer the topic from the chat history.
        - The rewritten query should be concise and suitable for legal document retrieval.

        ---

        Chat History:
        {chat_history}

        Original Query:
        {original_query}

        Rewritten Query:

"""

query_rewrite_prompt = ChatPromptTemplate.from_template(query_rewrite_template)
query_rewriter = query_rewrite_prompt | llm_together


# Classification prompt
classification_prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI assistant helping to classify user queries.

You must decide if the user's current message is a **legal query** or a **general/non-legal query**, based on both:
1. The current message
2. The previous conversation context (chat history)

---

Classify as **legal** if the query:
- Involves legal terminology, case law, legal processes, rights, or duties
- Asks about court decisions, case summaries, or legal outcomes
- Asks for summaries or analysis of legal documents or judgments
- Refers indirectly to earlier legal context, even if not explicitly repeated in the query

Classify as **general** if the query:
- Involves small talk, general knowledge, or day-to-day topics
- Asks unrelated things (jokes, stories, definitions, etc.)
- Does not relate to legal topics even with context

---

Chat history:
{chat_history}

Current user message:
{rewritten_query}

Answer in a single word: 'legal' or 'general'

""")

query_classifier = classification_prompt | llm_together

def format_chat_history(messages):
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages)

def process_query(original_query, chat_messages):
    """
    Rewrites a query using chat history and classifies it as legal or general.
    """
    chat_history = format_chat_history(chat_messages)
    
    # Step 1: Rewrite
    rewritten = query_rewriter.invoke({
        "original_query": original_query,
        "chat_history": chat_history
    }).content
    
    # Step 2: Classify
    classification = query_classifier.invoke({
        "rewritten_query": rewritten,
        "chat_history": chat_history
    }).content

    return {
        "rewritten_query": rewritten,
        "classification": classification
    }


