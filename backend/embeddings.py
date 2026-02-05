from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(pdf_path: str):
    """
    Loads a PDF, splits text into chunks, generates embeddings,
    and stores them in a FAISS vector store.
    """

    # 1️⃣ Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2️⃣ Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = text_splitter.split_documents(documents)

    # 3️⃣ Load Embedding Model (Offline, Fast, Free)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4️⃣ Create FAISS Vector Store
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    return vectorstore
