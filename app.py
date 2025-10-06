import streamlit as st
import os
from rag import setup_rag_chain, search_duckduckgo
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title='AnimeKIQ Chatbot', layout="centered", page_icon="🎌")
st.title("🎌 AnimeKIQ: Your Anime AI Chatbot")
st.caption("Enter an Anime URL, then chat naturally about it — powered by RAG + Groq Llama 3")
st.image("image.png", use_container_width=True)

# -------------------- Session State --------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    # LangChain Conversation Memory
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# -------------------- Anime URL Input --------------------
mal_url = st.text_input("🔗 Enter an Anime URL", key="mal_url_input")

if mal_url:
    if mal_url != st.session_state.processed_url:
        with st.spinner(f"Analyzing {mal_url} ... please wait!"):
            st.session_state.retriever = setup_rag_chain(mal_url)
            if st.session_state.retriever:
                st.success("✅ Anime URL processed successfully!")
                st.session_state.processed_url = mal_url
                st.session_state.messages = []
                st.session_state.memory.clear()  # reset conversation context
            else:
                st.error("❌ Failed to process URL. Please try again.")
                st.session_state.processed_url = ""
else:
    st.info("👆 Enter an Anime URL above to start chatting.")

# -------------------- Chat Interface --------------------
if st.session_state.retriever and st.session_state.processed_url == mal_url:
    st.divider()
    st.markdown("### 💬 Chat with AnimeKIQ")

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input from user
    if prompt := st.chat_input("Ask me anything about this anime..."):
        # Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve context from FAISS
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    rag_context = "\n".join([doc.page_content for doc in retrieved_docs])

                    # Web search context
                    duckduckgo_results = search_duckduckgo(prompt)

                    # Combine contexts
                    final_text = f"RAG DATA:\n{rag_context}\n\nWEB SEARCH:\n{duckduckgo_results}"

                    # Initialize LLM
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.8,
                        api_key=os.getenv("GROQ_API_KEY")
                    )

                    # Build prompt template
                    prompt_template = ChatPromptTemplate.from_template("""
                        You are AnimeKIQ, an AI anime expert.
                        Continue this conversation naturally, remembering previous context.
                        Use both the RAG data and web search results when relevant.
                        If unsure, say you don’t have enough data.
                        Cite the source as [RAG] or [Search].

                        PAST CONVERSATION:
                        {history}

                        CONTEXT:
                        {context}

                        USER QUESTION:
                        {question}

                        ANSWER:
                    """)

                    # Get conversation history from memory
                    history_str = "\n".join(
                        [f"{m.type.capitalize()}: {m.content}" for m in st.session_state.memory.chat_memory.messages]
                    )

                    # Format full prompt
                    formatted_prompt = prompt_template.format(
                        history=history_str, context=final_text, question=prompt
                    )

                    # Generate response
                    response = llm.invoke(formatted_prompt)
                    bot_message = response.content.strip()
                    st.markdown(bot_message)

                    # Save both user + bot to memory and session
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(bot_message)
                    st.session_state.messages.append({"role": "assistant", "content": bot_message})

                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
