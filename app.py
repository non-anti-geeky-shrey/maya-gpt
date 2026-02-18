import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Config & Universal Background Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

st.markdown("""
    <style>
    /* Setting a Universal Background Image */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("https://images.unsplash.com/photo-1464802686167-b939a67e06a1?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }
    
    /* Elegant Title */
    .main-title {
        text-align: center; 
        font-size: 3.5rem; 
        font-weight: 800;
        background: -webkit-linear-gradient(#ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Chat Bubble Styling */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.07) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        margin-bottom: 15px;
    }

    /* Input Bar Styling */
    .stChatInput input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. AI Engine Setup
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

# 4. Header
st.markdown('<h1 class="main-title">Maya-GPT</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity: 0.8;'>The Universal Bridge: Science & Consciousness</p>", unsafe_allow_html=True)
st.write("---")

# 5. Display Chat History
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧘"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 6. Chat Logic
if prompt := st.chat_input("Ask the Nexus..."):
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate Maya's Response
    with st.chat_message("assistant", avatar="🧘"):
        with st.spinner("Decoding the weave..."):
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            
            system_prompt = (
                "You are Maya-GPT, a synthesis of the world's deepest philosophies and cutting-edge physics. "
                "Use the following context to provide a profound and poetic bridge between Science and Vedanta.\n\n"
                f"Context: {context}\n\nQuestion: {prompt}"
            )
            
            response = llm.invoke(system_prompt)
            full_response = response.content
            st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
