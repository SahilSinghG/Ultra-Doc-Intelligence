import streamlit as st
import requests

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "uploaded_files_set" not in st.session_state:
    st.session_state.uploaded_files_set = set()

API_URL = "http://127.0.0.1:8000"

st.title("Ultra Doc Intelligence")

st.write("Upload a logistics document and ask questions about it.")

# Upload document
uploaded_files = st.file_uploader(
    "Upload document(s)",
    accept_multiple_files=True
)

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if uploaded_files:

    for file in uploaded_files:

        if file.name not in st.session_state.uploaded_files_set:

            files = {"file": (file.name, file, file.type)}

            response = requests.post(f"{API_URL}/upload", files=files)

            if response.status_code == 200:
                st.success(f"{file.name} uploaded successfully")

                st.session_state.uploaded_files_set.add(file.name)
                st.session_state.uploaded = True

# Ask question
question = st.text_input("Ask a question")

# ✅ ADD THIS CHECK HERE
if not st.session_state.get("uploaded", False):
    st.warning("Please upload a document first")

elif st.button("Ask"):

    response = requests.post(
        f"{API_URL}/ask",
        params={"question": question}
    )

    result = response.json()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Confidence")
    st.write(result["confidence"])

    st.subheader("Sources")
    for s in result["sources"]:
        st.text(s)

# Structured extraction
if st.button("Extract Shipment Data"):

    response = requests.post(f"{API_URL}/extract")

    result = response.json()

    st.subheader("Extracted Shipment Data")

    st.json(result)

if st.button("Clear Document"):

    requests.post(f"{API_URL}/reset")

    st.session_state.clear()

    st.success("Document cleared")

    st.rerun()