import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Premium Page Configuration
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

# Custom CSS for the "Nexus" Glassmorphism Look
st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
        color: #f8fafc; 
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important; 
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 14px !important;
        padding: 12px !important;
        font-size: 1.1rem;
    }
    .wisdom-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
        margin-top: 30px;
        font-size: 1.15rem;
        line-height: 1.7;
        color: #e2e8f0;
    }
    .maya-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar branding
with st.sidebar:
    st.markdown("## 🧘 Maya-GPT")
    st.caption("The Universal Bridge")
    st.markdown("---")
    st.write("Bridging the divide between modern science and ancient wisdom.")
    st.write("⚛️ Quantum Physics\n🧠 Consciousness\n📜 Eastern Philosophy\n🏛️ Western Philosophy")

# 3. AI Engine Setup
api_key = st.secrets.get("GROQ_API_KEY")

class SimpleEmbedder:
    def __init__(self): 
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, texts): 
        return self.model.encode(texts).tolist()
    def embed_query(self, text): 
        return self.model.encode([text])[0].tolist()

@st.cache_resource
def init_system():
    if not api_key:
        st.error("API Key missing in Secrets!")
        st.stop()
    embeddings = SimpleEmbedder()
    vectorstore = Chroma(persist_directory="./maya_db", embedding_function=embeddings)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    return vectorstore.as_retriever(search_kwargs={"k": 4}), llm

retriever, llm = init_system()

# 4. Main UI
st.markdown('<p class="maya-title">Maya-GPT</p>', unsafe_allow_html=True)
st.write("Synthesizing Quantum Physics, Consciousness, and Universal Philosophy.")

query = st.text_input("", placeholder="Ask the Nexus...", label_visibility="collapsed")

if query:
    with st.spinner("Decoding the weave..."):
        # Everything inside this 'if' block must be indented!
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_prompt = (
            "You are Maya-GPT, a synthesis of the world's deepest philosophies and cutting-edge physics. "
            "Use the provided context to bridge the gap between Quantum Physics, Consciousness, Vedanta, "
            "and Western Philosophy. Speak with the authority of a sage and the precision of a scientist."
            f"\n\nContext: {context}\n\nQuestion: {query}"
        )
        
        response = llm.invoke(system_prompt)
        
        # Display response in the premium card
        st.markdown(f'<div class="wisdom-card"><b>Maya\'s Insight:</b><br><br>{response.content}</div>', unsafe_allow_html=True)
