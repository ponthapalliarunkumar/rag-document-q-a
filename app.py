"""
RAG-based Document Q&A System — Streamlit Web App
---------------------------------------------------
Answers questions grounded in uploaded documents using
Retrieval-Augmented Generation.

Author: Ponthapalli Arun Kumar
"""

import os
import glob

import numpy as np
import faiss
import streamlit as st

from google import genai
from google.genai import types


# =========================================================
# Configuration
# =========================================================

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"

EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# Resolve paths relative to this script, not the current working directory.
# This avoids "No documents available" errors caused by CWD mismatches.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")


# =========================================================
# Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)


# =========================================================
# Gemini Client
# =========================================================

def get_gemini_client() -> genai.Client:
    """
    Create Gemini client using a secret/API key.

    Local:
        .streamlit/secrets.toml

    Streamlit Cloud:
        App Settings -> Secrets
    """

    api_key = (
        st.secrets.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    if not api_key:
        st.error(
            "No Google API key found. "
            "Add GOOGLE_API_KEY to Streamlit Secrets "
            "or set it as an environment variable."
        )
        st.stop()

    return genai.Client(api_key=api_key)


# =========================================================
# Text Chunking
# =========================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list:

    chunks = []

    if not text.strip():
        return chunks

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# Generate Embeddings
# =========================================================

def embed_texts(
    client: genai.Client,
    texts: list
) -> np.ndarray:

    if not texts:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype="float32"
        )

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    vectors = [
        embedding.values
        for embedding in result.embeddings
    ]

    return np.array(
        vectors,
        dtype="float32"
    )


# =========================================================
# Build FAISS Index
# =========================================================

@st.cache_resource(show_spinner=False)
def build_index_from_texts(
    _client: genai.Client,
    doc_texts: tuple
):

    all_chunks = []

    for filename, text in doc_texts:

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append(
                {
                    "source": filename,
                    "text": chunk
                }
            )

    if not all_chunks:
        raise ValueError(
            "No readable text was found in the documents."
        )

    # Generate embeddings
    vectors = embed_texts(
        _client,
        [chunk["text"] for chunk in all_chunks]
    )

    # Create FAISS index
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # Add vectors
    index.add(vectors)

    return index, all_chunks


# =========================================================
# Retrieve Relevant Chunks
# =========================================================

def retrieve(
    client: genai.Client,
    index,
    all_chunks,
    query: str,
    top_k: int = TOP_K
) -> list:

    query_vector = embed_texts(
        client,
        [query]
    )

    k = min(
        top_k,
        len(all_chunks)
    )

    distances, indices = index.search(
        query_vector,
        k
    )

    retrieved_chunks = []

    for index_position in indices[0]:

        if index_position != -1:

            retrieved_chunks.append(
                all_chunks[index_position]
            )

    return retrieved_chunks


# =========================================================
# Generate Grounded Answer
# =========================================================

def answer_question(
    client: genai.Client,
    index,
    all_chunks,
    query: str
) -> str:

    retrieved = retrieve(
        client,
        index,
        all_chunks,
        query
    )

    if not retrieved:
        return (
            "I don't have enough information "
            "in the provided documents."
        )

    context = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in retrieved
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the document
context provided below.

Rules:
1. Use only information from the provided context.
2. Do not use outside knowledge.
3. Do not guess or invent information.
4. If the answer is not present in the context, say:
   "I don't have enough information in the provided documents."
5. Keep the answer clear and concise.
6. Mention the source document when appropriate.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    if not response.text:
        return (
            "I couldn't generate an answer. "
            "Please try again."
        )

    return response.text.strip()


# =========================================================
# Load Sample Documents
# =========================================================

def load_sample_documents() -> list:

    docs = []

    os.makedirs(DOCS_DIR, exist_ok=True)

    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                content = file.read()

                if content.strip():

                    docs.append(
                        (
                            os.path.basename(path),
                            content
                        )
                    )

        except OSError as error:

            st.warning(
                f"Could not read {path}: {error}"
            )

    return docs


# =========================================================
# Streamlit UI
# =========================================================

def main():

    # Create Gemini client
    client = get_gemini_client()

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    st.title("📄 RAG Document Q&A")

    st.caption(
        "Ask questions grounded in real document content — "
        "powered by Retrieval-Augmented Generation."
    )

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    with st.sidebar:

        st.header("📚 Documents")

        uploaded_files = st.file_uploader(
            "Upload .txt files",
            type=["txt"],
            accept_multiple_files=True
        )

        st.caption(
            "If no files are uploaded, documents from "
            "`documents/` will be used."
        )

        with st.expander("🔧 Debug info"):
            st.write("**Script directory:**", BASE_DIR)
            st.write("**Documents directory:**", DOCS_DIR)
            st.write("**Folder exists?**", os.path.isdir(DOCS_DIR))
            if os.path.isdir(DOCS_DIR):
                st.write("**Files found:**", os.listdir(DOCS_DIR))

        st.divider()

        st.header("⚙️ Configuration")

        st.write(
            f"**Embedding:** `{EMBED_MODEL}`"
        )

        st.write(
            f"**Chat model:** `{CHAT_MODEL}`"
        )

        st.write(
            f"**Embedding dimension:** `{EMBEDDING_DIMENSION}`"
        )

        st.write(
            f"**Chunk size:** `{CHUNK_SIZE}`"
        )

        st.write(
            f"**Chunk overlap:** `{CHUNK_OVERLAP}`"
        )

        st.write(
            f"**Top K:** `{TOP_K}`"
        )

    # -----------------------------------------------------
    # Gather Documents
    # -----------------------------------------------------

    doc_texts = []

    if uploaded_files:

        for uploaded_file in uploaded_files:

            try:

                content = uploaded_file.read().decode(
                    "utf-8",
                    errors="ignore"
                )

                if content.strip():

                    doc_texts.append(
                        (
                            uploaded_file.name,
                            content
                        )
                    )

            except Exception as error:

                st.error(
                    f"Could not read {uploaded_file.name}: "
                    f"{error}"
                )

    else:

        doc_texts = load_sample_documents()

    # -----------------------------------------------------
    # Check Documents
    # -----------------------------------------------------

    if not doc_texts:

        st.warning(
            "No documents available. "
            "Upload a .txt file or add a document "
            "inside the documents/ folder."
        )

        st.stop()

    # -----------------------------------------------------
    # Build Vector Index
    # -----------------------------------------------------

    with st.spinner(
        "Generating embeddings and indexing documents..."
    ):

        try:

            index, all_chunks = build_index_from_texts(
                client,
                tuple(doc_texts)
            )

        except Exception as error:

            st.error(
                "Error while creating the document index."
            )

            st.exception(error)

            st.stop()

    # -----------------------------------------------------
    # Index Status
    # -----------------------------------------------------

    st.success(
        f"Indexed {len(doc_texts)} document(s) "
        f"into {len(all_chunks)} chunks."
    )

    # -----------------------------------------------------
    # Question
    # -----------------------------------------------------

    query = st.text_input(
        "Ask a question about the document(s):",
        placeholder=(
            "Example: How many annual leave days "
            "are available?"
        )
    )

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    if query.strip():

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer = answer_question(
                    client,
                    index,
                    all_chunks,
                    query.strip()
                )

            except Exception as error:

                st.error(
                    "Error while generating the answer."
                )

                st.exception(error)

                st.stop()

        st.markdown("### 🤖 Answer")

        st.write(answer)


# =========================================================
# Start Application
# =========================================================

if __name__ == "__main__":
    main()