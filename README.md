# 🎌 AnimeKIQ — Your Anime AI Chatbot

AnimeKIQ is an intelligent anime chatbot built using **Retrieval-Augmented Generation (RAG)** and **LangChain**, powered by **Groq Llama 3** for natural and context-aware conversations.  
This project marks my **first hands-on RAG implementation**, built to deepen my understanding of how language models can combine **vector-based retrieval**, **live web search**, and **LLM reasoning** to create knowledge-grounded chatbots.

---

## 🧠 Project Overview

AnimeKIQ is an **Anime-focused AI Chatbot** that allows users to:
- Enter any **Anime URL** (e.g., MyAnimeList or fandom pages)
- Automatically **scrape and process** the content for embeddings
- Combine retrieved knowledge with **live web search results**
- Chat naturally about that anime — characters, plot, reviews, and more.

The goal of this project was to **learn the fundamentals of RAG pipelines**, including:
- Text chunking and vector embeddings
- FAISS-based semantic search
- Real-time augmentation with external data
- Memory and context preservation in conversations

---

## ⚙️ How It Works (Workflow)

### **1. Input Stage**
Users input an anime URL (for example, a MyAnimeList link).  
The app extracts and processes the text content from that URL.

### **2. Text Processing**
Using **LangChain’s RecursiveCharacterTextSplitter**, the anime description is divided into smaller text chunks for better embedding performance.

### **3. Embedding & Storage**
The chunks are converted into **vector embeddings** using:
```python
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")```

These embeddings are stored in a FAISS vector database for fast semantic retrieval.


### **4. Query-Time Retrieval**
When the user asks a question:
- Relevant chunks are retrieved from FAISS (RAG context)
- Simultaneously, a live web search is performed using the ddgs (DuckDuckGo Search) API to fetch up-to-date results.
