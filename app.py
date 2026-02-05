import streamlit as st
import tempfile
from dotenv import load_dotenv

from backend.embeddings import create_vector_store
from backend.qa_chain import generate_answer

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="VTU AI Solver",
    page_icon="📘",
    layout="centered"
)

# App Header
st.title("📘 VTU Question Paper Answer Generator")
st.caption("Upload VTU notes or question paper PDF and generate exam-ready answers")

# File uploader
uploaded_file = st.file_uploader(
    "📄 Upload VTU PDF (Notes / Question Paper)",
    type=["pdf"]
)

# Main Logic
if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully")

    # Create Vector Store
    with st.spinner("🔍 Processing PDF and building knowledge base..."):
        vectorstore = create_vector_store(pdf_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    st.divider()

    # Question Input
    question = st.text_input(
        "✍️ Enter VTU Question",
        placeholder="Explain Greedy Best First Search"
    )

    # Marks Selection
    marks = st.selectbox(
        "🎯 Select Answer Length",
        ["2 Marks", "5 Marks", "10 Marks", "15 Marks"]
    )

    # Generate Answer Button
    if st.button("🚀 Generate VTU Answer", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Please enter a valid question.")
        else:
            with st.spinner("🧠 Generating VTU-style answer..."):
                docs = retriever.invoke(question)
                answer = generate_answer(
                    question=question,
                    docs=docs,
                    marks=marks
                )

            st.subheader("📝 VTU Exam Answer")
            st.write(answer)

else:
    st.info("⬆️ Upload a PDF to start generating answers.")
