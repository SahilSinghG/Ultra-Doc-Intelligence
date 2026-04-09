from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

VECTOR_DB_PATH = "vector_db"


def load_vector_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


import json

llm = ChatOllama(
    model="mistral",
    temperature=0
)

def extract_data(docs):

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Extract shipment details from the document.

Return ONLY valid JSON.
Do NOT add explanation.
Do NOT wrap in quotes.
Do NOT add extra text.

If a field is missing, return null.

Fields:
shipment_id, shipper, consignee, pickup_datetime,
delivery_datetime, equipment_type, mode,
rate, currency, weight, carrier_name

Context:
{context}
"""

    response = llm.invoke(prompt)

    raw_output = response.content.strip()

    try:
        parsed = json.loads(raw_output)
        return parsed
    except Exception as e:
        print("❌ JSON PARSE ERROR")
        print(raw_output)
        return {
            "error": "Invalid JSON from LLM",
            "raw_output": raw_output
        }