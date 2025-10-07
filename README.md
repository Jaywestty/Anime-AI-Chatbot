# AnimeKIQ — Your Anime AI Chatbot

**AnimeKIQ** is an intelligent anime chatbot built using **Retrieval-Augmented Generation (RAG)** and **LangChain**, powered by **Groq Llama 3** for natural and context-aware conversations.

This project marks my **first hands-on RAG implementation**, built to deepen my understanding of how language models can combine **vector-based retrieval**, **live web search**, and **LLM reasoning** to create knowledge-grounded chatbots.

---

## 🧠 Project Overview

**AnimeKIQ** is an **anime-focused AI chatbot** that allows users to:
* Enter any **Anime URL** (e.g., MyAnimeList, Fandom, or AnimePlanet pages).
* Automatically **scrape and process** the content for embeddings.
* Combine retrieved knowledge with **live web search results**.
* Chat naturally about that anime — characters, plots, reviews, and more.

The primary goal of this project was to **learn the fundamentals of RAG pipelines**, including:
* Text chunking and vector embeddings
* FAISS-based semantic search
* Real-time augmentation with external data
* Memory and context preservation in conversations

---

## ⚙️ How It Works (Workflow)

### 1. Input Stage
* Users input an anime URL (for example, a MyAnimeList or Fandom link).
* The app extracts and processes the text content from that URL.

### 2. Text Processing
* Using **LangChain’s `RecursiveCharacterTextSplitter`**, the anime text is divided into smaller, context-friendly chunks for efficient embedding and retrieval.

### 3. Embedding & Storage
* The chunks are converted into **vector embeddings** using the `sentence-transformers/all-MiniLM-L6-v2` model:
    ```python
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ```
* These embeddings are stored in a **FAISS vector database** for fast semantic retrieval.

### 4. Query-Time Retrieval
* When the user asks a question:
    * Relevant text chunks are retrieved from FAISS (RAG context).
    * Simultaneously, a live web search is performed using the **`ddgs` (DuckDuckGo Search) API** to fetch up-to-date anime-related results.
* Both sources are combined to form a context-aware prompt for the model.

### 5. LLM Reasoning
* The **Groq Llama 3** model generates intelligent responses grounded in retrieved knowledge and live search context.

---

## 🧩 Tech Stack & Dependencies

| Component | Technology |
| :--- | :--- |
| **LLM** | Groq Llama 3 |
| **Framework** | LangChain |
| **Vector DB** | FAISS |
| **Embeddings** | Sentence Transformers (`MiniLM-L6-v2`) |
| **Search API** | `ddgs` (DuckDuckGo Search) |
| **Backend** | Python |
| **Frontend / Deployment**| Streamlit |

---

## 🚀 Deployment

The app is fully deployed on Streamlit Cloud, making it accessible for interactive use.

👉 **Try it out here:** [Live App Link](https://animekiqchatbot.streamlit.app/)

---

## 📚 Learning Outcomes

This project helped me understand:
* The end-to-end RAG workflow.
* How to integrate LLMs with vector stores and external APIs.
* Techniques for scraping, chunking, and semantic search.
* The process of building and deploying an AI chatbot using Streamlit.

---

## 🧾 Requirements

Key dependencies used in this project:
langchain
langchain-community
groq
faiss-cpu
sentence-transformers
ddgs
beautifulsoup4
requests
streamlit
python-dotenv

---

## 📦 Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jaywestty/Anime-AI-Chatbot.git](https://github.com/Jaywestty/Anime-AI-Chatbot.git)
    cd animekiq
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run locally:**
    ```bash
    streamlit run app.py
    ```

---

## 💡 Future Improvements

* Add conversation memory for multi-turn dialogue.
* Integrate anime-specific APIs (e.g., MyAnimeList API).
* Support image or video-based context generation.

---

## ✨ Acknowledgments

* **LangChain** for RAG pipeline components
* **Groq** for fast inference
* **HuggingFace** for embeddings
* **Streamlit** for easy deployment

---

## 👨‍💻 Author

**Fadairo Oluwajuwon**
*Data Scientist & Machine Learning Engineer*

* 📧 **Email:** `juwonfadairo13@gmail.com`
* 🔗 **GitHub Profile:** [https://github.com/Jaywestty]

