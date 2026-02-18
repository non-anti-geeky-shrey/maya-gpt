import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Config & Symmetry Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

st.markdown("""
    <style>
    /* 1. Deep HD Background */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.88), rgba(0, 0, 0, 0.88)), 
                    url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=2022&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }
    
    /* 2. Perfectly Centered Header & Subtopic */
    .header-container {
        text-align: center;
        margin-top: 40px;
        margin-bottom: 50px;
    }

    .main-title {
        font-size: 4.2rem; 
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(180deg, #ffffff 0%, #64748b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
    }

    .subtopic {
        font-size: 0.85rem;
        letter-spacing: 6px;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 15px;
    }

    /* 3. Symmetric Glass Bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(25px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 25px !important;
        margin-bottom: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }

    /* 4. Balanced Input Field */
    .stChatInput {
        padding-bottom: 40px;
    }
    
    .stChatInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 15px !important;
    }

    /* Hide unnecessary UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. Session State for Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. AI Core Setup
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

# 4. Perfectly Centric Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Maya-GPT</h1>
        <p class="subtopic">Synthesis of Quantum Physics & Ancient Wisdom</p>
    </div>
    """, unsafe_allow_html=True)

# 5. Chat Display
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧘"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 6. Interaction Logic
if prompt := st.chat_input("Ask the Nexus..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧘"):
        with st.spinner("Decoding..."):
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            system_prompt = (
                "You are Maya-GPT, a profound synthesis of Science and Philosophy. "
                f"Context: {context}\n\nQuestion: {prompt}"
            )
            response = llm.invoke(system_prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
