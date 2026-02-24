# RAG-Resume
FAQ Bot is a Streamlit-based RAG application that lets users upload PDFs and ask questions in natural language. It chunks documents, stores embeddings in ChromaDB, retrieves relevant context via semantic search, and uses Gemini to generate accurate answers, with real-time token usage tracking.


FAQ Bot

An AI-powered Streamlit application that allows users to upload a PDF and ask questions about its contents using a Retrieval-Augmented Generation (RAG) pipeline. The app converts documents into semantic embeddings, retrieves relevant context, and generates accurate answers in a conversational interface.

Features

📄 PDF Upload — Upload any PDF document for instant processing

✂️ Auto Chunking — Automatically splits the document into searchable segments

🔍 Semantic Search — Retrieves relevant chunks using vector embeddings

💬 Chat Interface — Interactive conversational Q&A with context memory

📊 Token Counter — Monitor API/token usage in real time

Run
cd 03-projects/15_faq_bot
pip install -r requirements.txt
streamlit run app.py
How It Works

Working Process:-
1)Upload a PDF document

2)The app extracts text and splits it into chunks (500 characters, 50 overlap)

3)Each chunk is converted into embeddings and stored in ChromaDB

4)When a question is asked, semantic search retrieves the most relevant chunks

5)The LLM generates an answer using the retrieved context

Tech Stack
1)Streamlit — Frontend interface

2)Gemini — Embeddings and language model

3)ChromaDB — Vector database for semantic retrieval

4)PyPDF2 — PDF text extraction
