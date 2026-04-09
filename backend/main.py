from fastapi import FastAPI, UploadFile, File
import shutil
import os
from backend.document_processor import process_document
from backend.rag_pipeline import ask_question
from backend.extraction import extract_data
from backend.rag_pipeline import clear_vector_db

app = FastAPI()

UPLOAD_FOLDER = "uploaded_docs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Ultra Doc Intelligence API running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # ✅ Save new file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ IMPORTANT: Reset old vector DB
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    # ✅ Process only THIS document
    process_document(file_path)

    return {
        "status": "New document uploaded and processed",
        "filename": file.filename
    }


@app.post("/ask")
def ask(question: str):

    result = ask_question(question)

    return result


from backend.rag_pipeline import load_vector_db

@app.post("/extract")
def extract():

    vector_store = load_vector_db()

    docs = vector_store.similarity_search("", k=20)

    result = extract_data(docs)

    return result

@app.post("/reset")
def reset():

    import shutil
    import os

    # delete vector DB folder
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    # delete uploaded docs
    if os.path.exists("uploaded_docs"):
        shutil.rmtree("uploaded_docs")
        os.makedirs("uploaded_docs", exist_ok=True)

    # ✅ CLEAR MEMORY CACHE
    clear_vector_db()

    return {"status": "reset successful"}