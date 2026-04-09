from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama



VECTOR_DB_PATH = "vector_db"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = None

def load_vector_db():

    global vector_store

    if vector_store is None:

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.load_local(
            "vector_db",
            embeddings,
            allow_dangerous_deserialization=True
        )

    return vector_store

def clear_vector_db():
    global vector_store
    vector_store = None

# Load LLM once
llm = ChatOllama(
    model="mistral",
    temperature=0,
    keep_alive="10m"
)


def retrieve_chunks(question):

    vector_store = load_vector_db()

    docs = vector_store.similarity_search_with_score(question, k=8)

    for doc, score in docs:
        print("Score:", score)

    return docs


def generate_answer(question, docs):

    context = "\n\n".join([doc.page_content for doc, score in docs[:6]])

    prompt = f"""
You are an AI assistant for logistics documents.

IMPORTANT RULES:
- If multiple answers exist, return ALL of them.
- Do not guess or add information not in the document.
- If the answer is not present, respond with "Not found in document".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content.strip()

def calculate_confidence(docs):

    best_score = docs[0][1]   # FAISS distance

    confidence = 1 / (1 + best_score)

    return round(float(confidence), 2)


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

    answer = generate_answer(question, docs)
    sources = [doc.page_content for doc, score in docs]
    confidence = calculate_confidence(docs)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }

def guardrail_check(docs):

    scores = [score for doc, score in docs]

    best_score = scores[0]
    avg_score = sum(scores) / len(scores)

    print("Best distance:", best_score)
    print("Average distance:", avg_score)

    # reject if everything looks weak
    if best_score > 1.6 and avg_score > 1.7:
        return False

    return True

def answer_exists(question, docs):

    context = "\n\n".join([doc.page_content for doc, score in docs])

    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    prompt = f"""
You are verifying whether a document contains the answer to a question.

Answer ONLY with YES or NO.

Context:
{context}

Question:
{question}

Does the document contain the answer?
"""

    response = llm.invoke(prompt).content.strip().upper()

    return response == "YES"