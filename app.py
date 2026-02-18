import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Config & Dark Chat Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

st.markdown("""
    <style>
    /* Midnight Dark Background */
    .stApp { 
        background-color: #0b0e14; 
        color: #e2e8f0; 
    }
    
    /* Header Styling */
    .main-title {
        text-align: center; 
        font-size: 2.5rem; 
        font-weight: 700;
        background: -webkit-linear-gradient(#ffffff, #4F8BF9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 20px;
    }

    /* Chat Input at the bottom */
    .stChatInputContainer {
        padding-bottom: 20px;
        background-color: transparent !important;
    }

    .stChatInput input {
        background-color: #1a1f29 !important;
        color: white !important;
        border: 1px solid #30363d !important;
    }

    /* Customizing Chat Bubbles for Dark Mode */
    [data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 15px !important;
        margin-bottom: 12px;
    }

    /* Sidebar Dark Adjustment */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Initialize Session State (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. AI Engine Setup (Cached)
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

# 4. UI Header
st.markdown('<h1 class="main-title">Maya-GPT</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e;'>The Universal Bridge: Science & Wisdom</p>", unsafe_allow_html=True)

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Logic
if prompt := st.chat_input("Ask the Nexus..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Decoding the weave..."):
            # RAG Retrieval
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            
            system_prompt = (
                "You are Maya-GPT, a synthesis of the world's deepest philosophies and cutting-edge physics. "
                "Bridge Quantum Physics, Consciousness, Vedanta, and Western Philosophy. "
                "Provide a profound, clear answer based on the context and conversation history.\n\n"
                f"Context: {context}\n\nQuestion: {prompt}"
            )
            
            response = llm.invoke(system_prompt)
            full_response = response.content
            st.markdown(full_response)
    
    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
