from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage


def get_llm():
    return ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0.2
    )


def vtu_prompt(question: str, context: str, marks: str) -> str:
    return f"""
You are a VTU university exam answer generator.

Instructions:
- Write the answer strictly for {marks}
- Use simple academic English
- Be concise and exam-oriented
- If marks are 5 or more, use bullet points
- If marks are 10 or more, use headings and a short conclusion
- Avoid unnecessary explanations

Context (from notes):
{context}

Question:
{question}

Answer:
"""


def generate_answer(question: str, docs, marks: str) -> str:
    llm = get_llm()

    # ✅ FIX: Extract page_content safely
    if docs:
        context = "\n".join([doc.page_content for doc in docs])
    else:
        context = "No relevant context found."

    prompt = vtu_prompt(question, context, marks)

    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )
        return response.content

    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"
