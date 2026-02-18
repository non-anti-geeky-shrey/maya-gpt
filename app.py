import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os

# 1. Page Config & Professional Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.88), rgba(0, 0, 0, 0.88)), 
                    url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=2022&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }
    
    /* Logo and Header Symmetry */
    .header-container {
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
    }

    .logo-img {
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
    }

    .main-title {
        font-size: 3.5rem; 
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(180deg, #ffffff 0%, #64748b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .subtopic {
        font-size: 0.8rem;
        letter-spacing: 5px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-top: 10px;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
    }

    .stChatInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. Header with Logo
# Note: logo.png must be in your GitHub repo for this to work
st.markdown('<div class="header-container">', unsafe_allow_html=True)
if os.path.exists("logo.png"):
    st.image("logo.png", width=120)
else:
    # This shows a placeholder if you haven't uploaded the file yet
    st.markdown("✨", unsafe_allow_html=True) 

st.markdown('<h1 class="main-title">Maya-GPT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtopic">Universal Wisdom Nexus</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 3. Backend & Chat Logic (Kept the same for stability)
if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = st.secrets.get("GROQ_API_KEY")

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

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧘"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Inquire..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧘"):
        with st.spinner("Decoding..."):
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            system_prompt = f"You are Maya-GPT. Synthesis of Science/Wisdom. Context: {context}\n\nQuestion: {prompt}"
            response = llm.invoke(system_prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
