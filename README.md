# PDF-Insight-Retrieval-System
.

🚀 Local RAG System using LangChain, Hugging Face & ChromaDB

This project implements a local Retrieval-Augmented Generation (RAG) pipeline using LangChain, Hugging Face embeddings, and a lightweight Phi-3 Mini LLM.
It allows users to upload or load a PDF, retrieve the most relevant document chunks, and generate context-aware answers using a locally running model.

📌 Features

🔍 PDF loading & preprocessing using PyPDFLoader and RecursiveCharacterTextSplitter.

🧠 Semantic embeddings generated with sentence-transformers/all-mpnet-base-v2.

🗂️ ChromaDB local vector store with persistent storage.

🤖 Local LLM (Phi-3-mini-4k-instruct) integrated using HuggingFacePipeline for fast, offline generation.

🧾 Context-aware responses using LangChain's prompt templates and chain execution.

🔁 Automatically detects and loads existing ChromaDB or creates a new one.

🛠️ Tech Stack

Python

LangChain

ChromaDB

Hugging Face Transformers

Sentence Transformers

⚙️ Installation
1️⃣ Clone the repository
2️⃣ Create and activate virtual environment
3️⃣ Install dependencies
▶️ Usage
Run the RAG system:

Type your question related to the content inside ML.pdf.

The system will:

Load the PDF

Split into semantic chunks

Retrieve top-k relevant chunks

Pass context + question to your local LLM

Output a context-grounded answer
📈 Future Improvements

Add Streamlit UI for interactive chat

Support for multiple document uploads

Replace Phi-3 Mini with larger models (e.g., Llama-3 8B)

Add caching for faster retrieval

Add evaluation metrics (RAGAS, embeddings similarity score)
Phi-3 Mini LLM

LangChain Community Libraries
