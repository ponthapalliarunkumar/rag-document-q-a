# 📄 RAG Document Q&A

A web-based Retrieval-Augmented Generation (RAG) application that allows users to upload text documents and ask questions based on their content.

The application uses Google Gemini for semantic embeddings and answer generation, FAISS for vector similarity search, and Streamlit for the interactive web interface.

---

## 🚀 Live Demo

👉 https://arun-rag-document.streamlit.app/

Try the application directly in your browser without installing anything.

---

## 📌 Project Overview

The RAG Document Q&A system answers questions using information retrieved from uploaded documents.

Instead of asking the language model to answer from general knowledge, the application follows a Retrieval-Augmented Generation workflow:

1. Load document content.
2. Split documents into smaller overlapping chunks.
3. Generate embeddings for each document chunk.
4. Store the embeddings in a FAISS vector index.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant document chunks.
7. Send the retrieved context to Google Gemini.
8. Generate an answer grounded in the retrieved document content.

This approach helps the application provide answers based on the actual document content rather than guessing.

---

## ✨ Features

- 📄 Upload `.txt` documents
- 📚 Built-in sample documents
- ✂️ Automatic document chunking
- 🔢 Gemini semantic embeddings
- 🔎 FAISS vector similarity search
- 🤖 Gemini-powered answer generation
- 🎯 Context-grounded responses
- 🔐 Secure API key management
- 🌐 Streamlit web interface
- ⚡ Cached document index
- 📊 Displays indexed document and chunk information

---

## 🧠 RAG Architecture

```text
                 USER
                   │
                   ▼
          Upload / Select Document
                   │
                   ▼
            Document Text
                   │
                   ▼
             Text Chunking
                   │
                   ▼
        Gemini Embedding Model
                   │
                   ▼
            Vector Embeddings
                   │
                   ▼
             FAISS Index
                   │
        ┌──────────▼──────────┐
        │   User Question     │
        └──────────┬──────────┘
                   │
                   ▼
        Gemini Question Embedding
                   │
                   ▼
          FAISS Similarity Search
                   │
                   ▼
        Top Relevant Document Chunks
                   │
                   ▼
          Retrieved Context
                   │
                   ▼
        Gemini Generative Model
                   │
                   ▼
             Final Answer
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Gemini API key either as an environment variable:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

or in `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

On **Streamlit Community Cloud**, add `GOOGLE_API_KEY` under
**App Settings → Secrets** instead.

## 📁 Project Structure

```
rag-document-q-a/
├── app.py
├── requirements.txt
├── documents/
│   ├── sample.txt
│   └── leave_policy.txt
├── .gitignore
└── README.md
```

## Contact

For questions or feedback, reach out at: ponthapalliarun@gmail.com