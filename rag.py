#Import required libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ddgs import DDGS


#load api key
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#create search function
def search_duckduckgo(query, num_results=3):
    """Perform a DuckDuckGo text search using ddgs package."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                # Safely extract body or title depending on structure
                snippet = r.get("body") or r.get("title") or ""
                results.append(snippet)
        return "\n".join(results)
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return "DuckDuckGo search failed."
    
#setup ragchain
def setup_rag_chain(url):
    """loads data, create embeddings, load vectorstore and retriever"""

    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store.as_retriever(search_kwargs={"k":5})
    except Exception as e:
        print(f"Error during RAG setup: {e}")
        return None

