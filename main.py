import streamlit as st
from groq import Groq
import chromadb
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# --------------------
# Load Environment Variables
# --------------------

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)



embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma = chromadb.Client()

try:
    collection = chroma.get_collection("study_notes")
except:
    collection = chroma.create_collection("study_notes")

# --------------------
# Functions
# --------------------

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text


def split_text(text, chunk_size=300):
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def store_chunks(chunks):

    try:
        collection.delete(
            ids=[str(i) for i in range(len(chunks))]
        )
    except:
        pass

    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks
    )


def retrieve(query):

    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    return results["documents"][0]


def ask_groq(question):

    context = " ".join(retrieve(question))

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
                Answer only using the provided context.
                If the answer is not found in the context,
                reply: Not found in uploaded PDF.
                """
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {question}
                """
            }
        ]
    )

    return response.choices[0].message.content


# --------------------
# Streamlit UI
# --------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions from it.")

if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file:

    if st.button("Process PDF"):

        with st.spinner("Reading PDF..."):
            text = read_pdf(uploaded_file)

        chunks = split_text(text)

        with st.spinner("Creating Embeddings..."):
            store_chunks(chunks)

        st.session_state.processed = True

        st.success("✅ PDF Processed Successfully!")

if st.session_state.processed:

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("Get Answer"):

        if question.strip():

            with st.spinner("Generating Answer..."):
                answer = ask_groq(question)

            st.subheader("Answer")
            st.write(answer)

        else:
            st.warning("Please enter a question.")