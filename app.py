import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page & UI Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="wide")

# Custom CSS for a "Glassmorphism" effect and high-end feel
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: white; }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white; border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px; padding: 10px;
    }
    .wisdom-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    .source-tag { font-size: 0.8rem; color: #94a3b8; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar with Custom Branding
with st.sidebar:
    st.title("💠 Maya-GPT")
    st.markdown("---")
    st.write("🌌 **Quantum Physics**")
    st.write("🧠 **Consciousness**")
    st.write("📜 **Ancient Vedanta**")
    st.write("🎭 **Modern Philosophy**")
    st.divider()
    st.button("Reset Nexus", on_click=lambda: st.rerun())

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
    st.error("Connect your Groq Key in Secrets!")
    st.stop()

retriever, llm = init_nexus()

# 4. Main Interface
st.title("🧘 Maya-GPT: The Universal Bridge")
st.write("Exploring the intersection of objective reality and subjective experience.")

query = st.text_input("Pose your question to the collective wisdom:", placeholder="e.g. Is consciousness fundamental to the universe?")

if query:
    with st.spinner("Decoding the weave..."):
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = (
            "You are Maya-GPT, a synthesis of the world's deepest philosophies and cutting-edge physics. "
            f"Context: {context}\n\nQuestion: {query}\n\n"
            "Provide a profound, structured answer that bridges these worlds. No disclaimers."
        )
        
        res = llm.invoke(prompt)
        st.markdown(f'<div class="wisdom-card"><b>Maya\'s Insight:</b><br><br>{res.content}</div>', unsafe_allow_html=True)

        with st.expander("🔍 View Retreived Fragments"):
            for d in docs: st.caption(f"• {d.page_content}")
