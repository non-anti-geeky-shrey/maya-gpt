import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Config & Ultra-Fluid Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

st.markdown("""
    <style>
    /* 1. Ultra-Slow Deep Space Pan */
    @keyframes slowPan {
        0% { background-position: 0% 50%; transform: scale(1); }
        50% { background-position: 100% 50%; transform: scale(1.05); }
        100% { background-position: 0% 50%; transform: scale(1); }
    }

    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), 
                    url("https://images.unsplash.com/photo-1464802686167-b939a67e06a1?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        /* Increased to 60 seconds for a very slow, subtle feel */
        animation: slowPan 60s ease-in-out infinite;
        color: #ffffff;
    }
    
    /* 2. Meditative Title Pulse */
    @keyframes slowBreath {
        0% { opacity: 0.8; text-shadow: 0 0 10px rgba(255,255,255,0.1); }
        50% { opacity: 1; text-shadow: 0 0 25px rgba(79, 139, 249, 0.4); }
        100% { opacity: 0.8; text-shadow: 0 0 10px rgba(255,255,255,0.1); }
    }

    .main-title {
        text-align: center; 
        font-size: 3.5rem; 
        font-weight: 800;
        background: -webkit-linear-gradient(#ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        /* 10 second cycle for a calm 'breath' */
        animation: slowBreath 10s ease-in-out infinite;
    }

    /* 3. Fluid Message Entrance with 'Settle' Effect */
    @keyframes fluidEntrance {
        from { 
            opacity: 0; 
            transform: translateY(30px) scale(0.98);
            filter: blur(5px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1);
            filter: blur(0px);
        }
    }

    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px !important;
        margin-bottom: 20px;
        /* Slower 0.8s entrance with a 'settling' curve */
        animation: fluidEntrance 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    /* Softening the Spinner */
    .stSpinner > div {
        border-top-color: #4F8BF9 !important;
    }

    /* Input Styling */
    .stChatInput input {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 30px !important;
        color: white !important;
        transition: all 0.3s ease;
    }
    
    .stChatInput input:focus {
        border: 1px solid rgba(79, 139, 249, 0.5) !important;
        background-color: rgba(255, 255, 255, 0.12) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. AI Setup
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

# 4. Content
st.markdown('<h1 class="main-title">Maya-GPT</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity: 0.6; letter-spacing: 3px; font-weight: 300;'>THE UNIVERSAL BRIDGE</p>", unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧘"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask the Nexus..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧘"):
        with st.spinner(""): # Empty spinner for cleaner look
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            
            system_prompt = (
                "You are Maya-GPT, a synthesis of the world's deepest philosophies and cutting-edge physics. "
                f"\n\nContext: {context}\n\nQuestion: {prompt}"
            )
            
            response = llm.invoke(system_prompt)
            full_response = response.content
            st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
