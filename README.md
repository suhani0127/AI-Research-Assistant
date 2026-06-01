# 📚 AI Research Assistant (RAG System)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions about their content.

## Features

- PDF Upload and Processing
- Text Extraction using PyPDF
- Document Chunking
- Semantic Embeddings
- FAISS Vector Search
- Context-Aware Question Answering
- Source Attribution
- Groq-powered LLM Responses

## Tech Stack

### Libraries
- Streamlit
- Sentence Transformers
- FAISS
- PyPDF
- Groq
- NumPy

### Workflow

PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
FAISS Retrieval
↓
Groq LLM
↓
AI Answer

## Run Locally

pip install -r requirements.txt

python -m streamlit run app.py