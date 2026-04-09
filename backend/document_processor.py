import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber

VECTOR_DB_PATH = "vector_db"


# ✅ Clean text (safe version)
def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())


# ✅ Better PDF extraction using pdfplumber
def load_pdf_with_pdfplumber(file_path):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    return text


# ✅ Main processing function
def process_document(file_path):

    print("Loading document...")

    # 🔹 Handle different file types
    if file_path.endswith(".pdf"):
        raw_text = load_pdf_with_pdfplumber(file_path)
        docs = [Document(page_content=raw_text)]

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
        docs = loader.load()

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

    else:
        raise ValueError("Unsupported file type")

    # ✅ Clean text
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    print("Chunking document...")

    # ✅ Better chunking for RAG
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    # ✅ Save vector DB
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)

    vector_store.save_local(VECTOR_DB_PATH)

    print("Document processed and stored in vector DB")