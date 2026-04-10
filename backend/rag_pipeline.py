import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

VECTOR_DB_PATH = "vector_db"

# Global cache
vector_store = None

# Load LLM once
llm = ChatOllama(
    model="mistral",
    temperature=0,
    keep_alive="10m"
)


# -----------------------------
# LOAD VECTOR DB (SAFE)
# -----------------------------
def load_vector_db():
    global vector_store

    if not os.path.exists(VECTOR_DB_PATH):
        raise ValueError("No document uploaded yet")

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return vector_store


# -----------------------------
# CLEAR VECTOR DB CACHE
# -----------------------------
def clear_vector_db():
    global vector_store
    vector_store = None


# -----------------------------
# RETRIEVE CHUNKS
# -----------------------------
def retrieve_chunks(question):
    vector_store = load_vector_db()

    docs = vector_store.similarity_search_with_score(question, k=6)

    for doc, score in docs:
        print("Score:", score)

    return docs


# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(question, docs):
    context = "\n\n".join([doc.page_content for doc, score in docs[:4]])

    prompt = f"""
You are an AI assistant for logistics documents.

RULES:
- Answer ONLY using the context.
- If multiple values exist, return ALL.
- Do NOT guess.
- If not found, say: "Not found in document".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content.strip()


# -----------------------------
# CONFIDENCE SCORE
# -----------------------------
def calculate_confidence(docs):
    scores = [score for _, score in docs[:3]]

    confidence = sum([1 / (1 + s) for s in scores]) / len(scores)

    return round(float(confidence), 2)


# -----------------------------
# GUARDRAIL 1: SIMILARITY CHECK
# -----------------------------
def guardrail_check(docs):
    scores = [score for _, score in docs]

    best_score = scores[0]
    avg_score = sum(scores) / len(scores)

    print("Best distance:", best_score)
    print("Average distance:", avg_score)

    if best_score > 1.6 and avg_score > 1.7:
        return False

    return True


# -----------------------------
# GUARDRAIL 2: ANSWER EXISTS
# -----------------------------
def answer_exists(question, docs):
    context = "\n\n".join([doc.page_content for doc, _ in docs[:4]])

    prompt = f"""
You are verifying whether the answer exists in the document.

Answer ONLY with YES or NO.

Context:
{context}

Question:
{question}

Does the document contain the answer?
"""

    response = llm.invoke(prompt).content.strip().upper()

    return response == "YES"


# -----------------------------
# MAIN RAG FUNCTION
# -----------------------------
def ask_question(question):

    try:
        docs = retrieve_chunks(question)
    except:
        return {
            "answer": "Please upload a document first",
            "sources": [],
            "confidence": 0
        }

    if not docs:
        return {
            "answer": "Not found in document",
            "sources": [],
            "confidence": 0
        }

    # Guardrail 1
    if not guardrail_check(docs):
        return {
            "answer": "Not found in document",
            "sources": [],
            "confidence": calculate_confidence(docs)
        }

    # Guardrail 2
    if not answer_exists(question, docs):
        return {
            "answer": "Not found in document",
            "sources": [],
            "confidence": calculate_confidence(docs)
        }

    answer = generate_answer(question, docs)

    sources = [doc.page_content for doc, _ in docs]
    confidence = calculate_confidence(docs)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }
