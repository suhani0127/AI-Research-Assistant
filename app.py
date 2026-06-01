import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq



load_dotenv()

client = Groq(api_key= st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title= "AI Research Assistant", layout="wide")

st.title("AI Research Assistant")

st.caption("Upload PDFs and ask questions using RAG")

st.success("Powered by FAISS + Sentence Transformers + Groq")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 100)

    chunks = splitter.split_text(text)

    model = SentenceTransformer("all-MiniLm-L6-v2")

    embeddings = model.encode(chunks)

    st.subheader("Embedding Shape")

    st.write(embeddings.shape)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    st.success(f"Stored {index.ntotal} chunks in FAISS")

    question = st.text_input("Ask a question about the PDF")

    if question:

        question_embedding = model.encode([question])

        distances, indices = index.search(np.array(question_embedding), k=3)

        context = ""

        retrieved_chunks = []

        for idx in indices[0]:

            retrieved_chunks.append(chunks[idx])

            context += chunks[idx]
            context += "\n\n"

        prompt = f"""
        You are an AI research assistant.

        Answer only using the provided context.

        If the answer is not present in the context, say:
        'I could not find this infromation in the uploaded document.'


        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = client.chat.completions.create(model = "llama-3.3-70b-versatile", messages=[{
            "role": "user",
            "content": prompt}],
            temperature= 0.3)
        

        st.subheader(" AI Answer")

        st.write(response.choices[0].message.content)

        with st.expander("View Sources"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"### Source {i}")

                st.write(chunk)
        
    st.success("PDF uploaded successfully!")

    st.subheader("Number of chunks")
    st.write(len(chunks))

    st.subheader("First chunk")
    st.write(chunks[0])

    st.subheader("PDF Preview")

    st.write(text[:3000])


