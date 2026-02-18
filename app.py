import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Config & Professional Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

# Custom CSS for a clean, modern look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stTextInput > div > div > input { border-radius: 10px; border: 1px solid #4F8BF9; }
    .response-container { 
        background-color: #161b22; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 5px solid #4F8BF9;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar Info
with st.sidebar:
    st.title("🧘 Maya-GPT")
    st.markdown("### The Universal Bridge")
    st.write("Synthesizing insights from:")
    st.write("⚛️ **Quantum Physics**")
    st.write("🧠 **Consciousness**")
    st.write("📜 **Eastern Philosophy**")
    st.write("🏛️ **Western Philosophy**")
    st.divider()
    st.caption("Powered by Llama 3.3 & Groq")

# 3. Core AI Setup
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

class SimpleEmbedder:
    def __init__(self): self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, texts): return self.model.encode(texts).tolist()
    def embed_query(self, text): return self.model.encode([text])[0].tolist()

@st.cache_resource
def load_assets():
    embeddings = SimpleEmbedder()
    vectorstore = Chroma(persist_directory="./maya_db", embedding_function=embeddings)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    return vectorstore.as_retriever(search_kwargs={"k": 4}), llm

retriever, llm = load_assets()

# 4. Main UI
st.title("🧘 Maya-GPT")
st.write("Ask a question to bridge the gap between ancient wisdom and modern science.")

user_query = st.text_input("", placeholder="Type your inquiry here...", label_visibility="collapsed")

if user_query:
    with st.spinner("Meditating on the data..."):
        # RAG Logic
        docs = retriever.invoke(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_prompt = (
            "You are Maya-GPT, a universal bridge between Quantum Physics, Consciousness, "
            "Eastern Philosophy (Vedanta), and Western Philosophy. Use the following context "
            "to provide a deep, poetic, yet scientifically grounded answer.\n\n"
            f"Context: {context}\n\n"
            f"User Question: {user_query}\n\n"
            "Answer directly and wisely without disclaimers about missing context."
        )
        
        response = llm.invoke(system_prompt)
        
        # Displaying the response in our custom CSS box
        st.markdown(f'<div class="response-container">{response.content}</div>', unsafe_allow_html=True)

        with st.expander("🔍 Source Wisdom (Retrieval Chunks)"):
            for i, doc in enumerate(docs):
                st.info(f"**Source {i+1}:** {doc.page_content}")
