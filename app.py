import streamlit as st
import os
from rag import setup_rag_chain, search_duckduckgo
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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
    st.session_state.messages = []  # store chat history

# -------------------- Anime URL Input --------------------
mal_url = st.text_input("🔗 Enter an Anime URL", key="mal_url_input")

if mal_url:
    # only reprocess if a new URL is provided
    if mal_url != st.session_state.processed_url:
        with st.spinner(f"Analyzing {mal_url} ... please wait!"):
            st.session_state.retriever = setup_rag_chain(mal_url)
            if st.session_state.retriever:
                st.success("✅ Anime URL processed successfully!")
                st.session_state.processed_url = mal_url
                st.session_state.messages = []  # reset chat
            else:
                st.error("❌ Failed to process URL. Please try again.")
                st.session_state.processed_url = ""
else:
    st.info("👆 Enter an Anime URL above to start chatting.")

# -------------------- Chat Interface --------------------
if st.session_state.retriever and st.session_state.processed_url == mal_url:
    st.divider()
    st.markdown("### 💬 Chat with AnimeKIQ")

    # display existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # user input
    if prompt := st.chat_input("Ask me anything about this anime..."):
        # add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # generate model response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Step 1: Retrieve context from FAISS
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    rag_context = "\n".join([doc.page_content for doc in retrieved_docs])

                    # Step 2: Get DuckDuckGo context
                    duckduckgo_results = search_duckduckgo(prompt)

                    # Step 3: Combine contexts
                    final_text = f"RAG DATA:\n{rag_context}\n\nWEB SEARCH:\n{duckduckgo_results}"

                    # Step 4: Initialize LLM
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.8,
                        api_key=os.getenv("GROQ_API_KEY")
                    )

                    # Step 5: Create prompt
                    prompt_template = ChatPromptTemplate.from_template("""
                        You are AnimeKIQ, an AI anime expert.
                        Use both the RAG data and web search results to answer questions clearly.
                        Respond naturally, in a friendly and conversational tone.
                        When possible, cite the source using [RAG] or [Search].
                        If you don’t find relevant info, say so politely.

                        CONTEXT:
                        {context}

                        USER QUESTION:
                        {question}

                        ANSWER:
                    """)
                    formatted_prompt = prompt_template.format(context=final_text, question=prompt)

                    # Step 6: Generate and display response
                    response = llm.invoke(formatted_prompt)
                    bot_message = response.content.strip()
                    st.markdown(bot_message)

                    # Step 7: Save to session
                    st.session_state.messages.append({"role": "assistant", "content": bot_message})

                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
