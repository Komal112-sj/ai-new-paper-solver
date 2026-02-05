# backend/llm.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load .env automatically (for local testing only)
load_dotenv()


def get_llm():
    """
    Initialize ChatGroq LLM using cloud API.
    Uses GROQ_API_KEY from environment variables or Streamlit secrets.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ GROQ_API_KEY not found. Set it in .env or Streamlit secrets.")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=api_key,
        host="https://api.groq.ai"  # ensures cloud API, not localhost
    )


def vtu_prompt(question: str, context: str, marks: str) -> str:
    """
    Format the question, context, and marks into a VTU exam-style prompt.
    """
    return f"""
You are a VTU university exam answer generator.

Instructions:
- Write strictly for {marks} marks
- Use simple academic English
- Be concise and exam-oriented
- If marks ≥ 5 → use bullet points
- If marks ≥ 10 → use headings and conclusion
- Avoid unnecessary explanations

Context:
{context}

Question:
{question}

Answer:
"""


def generate_answer(question: str, docs, marks: str) -> str:
    """
    Generate answer using ChatGroq LLM for a given question and context documents.
    """
    try:
        llm = get_llm()
    except Exception as e:
        return f"⚠️ LLM initialization error: {str(e)}"

    # Combine context documents into one string
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
    prompt = vtu_prompt(question, context, marks)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"
