import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Config & Modern Chat Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #fdfcfb; color: #333333; }
    
    /* Styling the Chat Input at the bottom */
    .stChatInputContainer {
        padding-bottom: 20px;
        background-color: transparent !important;
    }

    /* Customizing Chat Bubbles */
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #f0f0f0 !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        margin-bottom: 10px;
    }
    
    .main-title {
        text-align: center; font-size: 2.2rem; font-weight: 300;
        color: #2d3436; margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Initialize Session State (This is the "Memory")
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores the chat history

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
st.markdown("<p style='text-align:center; color:#636e72;'>The Universal Bridge</p>", unsafe_allow_html=True)

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input Logic
if prompt := st.chat_input("Share a thought or ask a question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Reflecting..."):
            # RAG Retrieval
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            
            system_prompt = (
                "You are Maya-GPT, a wise and empathetic guide. "
                "Synthesize Quantum Physics and Philosophy based on the context. "
                "If the history is relevant, use it to build the conversation.\n\n"
                f"Context: {context}\n\nQuestion: {prompt}"
            )
            
            response = llm.invoke(system_prompt)
            full_response = response.content
            st.markdown(full_response)
    
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
