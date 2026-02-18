import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page & UI Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="wide")

# Enhanced "Premium Universal Bridge" Styling
st.markdown("""
    <style>
    .main { 
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
        color: white; 
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.07) !important;
        color: white !important; 
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    .wisdom-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
        margin-top: 25px;
        line-height: 1.6;
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 10px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar with Custom Branding (Reset Button Removed)
with st.sidebar:
    st.title("💠 Maya-GPT")
    st.markdown("### The Universal Bridge")
    st.write("Synthesizing insights from:")
    st.write("🌌 **Quantum Physics**")
    st.write("🧠 **Consciousness**")
    st.write("📜 **Ancient Vedanta**")
    st.write("🎭 **Modern Philosophy**")
    st.divider()
    st.caption("Llama 3.3 | Groq LPU™")

# 3. Backend (Your RAG Logic)
api_key = st.secrets.get("GROQ_API_KEY")

class SimpleEmbedder:
    def __init__(self): self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, texts): return self.model.encode(texts).tolist()
    def embed_query(self, text): return self.model.encode([text])[0].tolist()

@st.cache_resource
def init_nexus():
    embeddings = SimpleEmbedder()
    vectorstore = Chroma(persist_directory="./maya_db", embedding_function=embeddings)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    return vectorstore.as_retriever(search_kwargs={"k": 4}), llm

if not api_key:
    st.error("Missing API Key! Please check Streamlit Secrets.")
    st.stop()

retriever, llm = init_nexus()

# 4. Main Interface
st.title("🧘 Maya-GPT: The Universal Bridge")
st.write("Exploring the intersection of objective reality and subjective experience.")

query = st.text_input("Pose your question to the collective wisdom:", placeholder="e.g. How does the observer effect relate to non-duality?")

if query:
    with st.spinner("Decoding the weave..."):
        # Retrieval
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Synthesis Prompt
        prompt = (
            "You are Maya-GPT, a synthesis of the world's deepest philosophies and cutting-edge physics. "
            "Your task is to bridge the gap between Quantum Physics, Consciousness, Vedanta, and Western Philosophy. "
            f"\n\nContext: {context}\n\nQuestion: {query}\n\n"
            "Provide a structured, profound answer. Do not use disclaimers like 'based on the context.' "
            "Speak with clarity and wisdom."
        )
        
        # LLM Call
        res = llm.invoke(prompt)
        
        # The beautiful response card
        st.markdown(f'<div class="wisdom-card"><b>Maya\'s Insight:</b><br><br>{res.content}</div>', unsafe_allow_html=True)

        # Subtle source expander
        with st.expander("🔍 View Retreived Fragments"):
            for d in docs: 
                st.caption(f"• {d.page_content}")
