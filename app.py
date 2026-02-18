import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="wide")

# Custom CSS to "beautify" the interface
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #ffffff;
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #4F8BF9;
        color: white;
    }
    .response-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4F8BF9;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar for Navigation & Info
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d47353039331dbf119627.svg", width=50)
    st.title("About Maya-GPT")
    st.info("""
    This AI bridges the gap between:
    * **Quantum Physics**
    * **Consciousness**
    * **Eastern Philosophy**
    * **Western Philosophy**
    """)
    st.divider()
    st.write("🔍 **Status:** Connected to 701 Wisdom Chunks")
    if st.button("Clear Conversation"):
        st.rerun()

# 3. Security
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Missing API Key!")
    st.stop()

# 4. Core Logic (Cached for speed)
class SimpleEmbedder:
    def __init__(self): self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, texts): return self.model.encode(texts).tolist()
    def embed_query(self, text): return self.model.encode([text])[0].tolist()

@st.cache_resource
def init_system():
    embeddings = SimpleEmbedder()
    vectorstore = Chroma(persist_directory="./maya_db", embedding_function=embeddings)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    return vectorstore.as_retriever(search_kwargs={"k": 4}), llm

retriever, llm = init_system()

# 5. Main UI
st.title("🧘 Maya-GPT: The Universal Bridge")
st.caption("Synthesizing ancient wisdom and modern science.")

user_query = st.text_input("Enter your inquiry:", placeholder="e.g., How does the observer effect relate to non-duality?")

if user_query:
    with st.spinner("Analyzing dimensions..."):
        docs = retriever.invoke(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_prompt = (
            f"You are Maya-GPT... (your prompt logic here)"
            f"Context: {context}\nQuestion: {user_query}"
        )
        
        response = llm.invoke(system_prompt)
        
        # Displaying response in a beautiful custom box
        st.markdown(f'<div class="response-box"><b>Maya\'s Insight:</b><br><br>{response.content}</div>', unsafe_allow_html=True)

        with st.expander("📚 View Underlying Data Chunks"):
            for doc in docs:
                st.write(f"- {doc.page_content}")
