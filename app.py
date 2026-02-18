import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Maya-GPT", 
    page_icon="Gemini_Generated_Image_vj0o5qvj0o5qvj0o.png", 
    layout="centered"
)

# 2. ADVANCED CSS: SYMMETRIC POSITIONING & HD GLASS
st.markdown("""
    <style>
    /* HD Background */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.88), rgba(0, 0, 0, 0.88)), 
                    url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=2022&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }

    /* Universal Chat Bubble Styling */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(25px) saturate(180%);
        -webkit-backdrop-filter: blur(25px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 15px !important;
        margin-bottom: 20px;
        width: fit-content !important;
        max-width: 85% !important;
    }

    /* POSITIONING: AI (Assistant) stays Left */
    .stChatMessage:has([aria-label="assistant"]) {
        margin-right: auto !important;
    }

    /* POSITIONING: User moves to Right */
    .stChatMessage:has([aria-label="user"]) {
        margin-left: auto !important;
        flex-direction: row-reverse !important;
        background: rgba(79, 139, 249, 0.08) !important; /* Subtle blue tint */
        border: 1px solid rgba(79, 139, 249, 0.2) !important;
    }

    /* Header Styling */
    .header-container { text-align: center; margin-top: 20px; margin-bottom: 40px; }
    .main-title {
        font-size: 3.8rem; font-weight: 900; letter-spacing: -2px;
        background: linear-gradient(180deg, #ffffff 0%, #64748b 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .subtopic { 
        font-size: 0.85rem; 
        letter-spacing: 4px; 
        text-transform: uppercase; 
        color: #94a3b8; 
        font-weight: 600;
        margin-top: 15px;
    }

    /* Input Bar Polish */
    .stChatInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 14px !important;
        color: white !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. BACKEND INITIALIZATION
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

# 4. CENTRIC HEADER
st.markdown('<div class="header-container">', unsafe_allow_html=True)
logo_path = "Gemini_Generated_Image_vj0o5qvj0o5qvj0o.png"
if os.path.exists(logo_path):
    left_co, cent_co, last_co = st.columns([1, 1, 1])
    with cent_co: st.image(logo_path, use_container_width=True)
st.markdown('<h1 class="main-title">Maya-GPT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtopic">Synthesis of Quantum Physics & Ancient Wisdom</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 5. CHAT DISPLAY (With Symmetric Avatars)
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧘"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 6. INTERACTION LOGIC
if prompt := st.chat_input("Ask the Nexus..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧘"):
        with st.spinner("Analyzing..."):
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            system_prompt = (
                "You are Maya-GPT. Provide a profound synthesis of Science and Wisdom. "
                f"Context: {context}\n\nQuestion: {prompt}"
            )
            response = llm.invoke(system_prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "Built with 💙 by Shrey Sharma | Powered by Llama 3.3 & RAG Logic"
    "</div>", 
    unsafe_allow_html=True
)
