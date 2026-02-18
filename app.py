import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Page Config & Deepak-Inspired Styling
st.set_page_config(page_title="Maya-GPT", page_icon="🧘", layout="centered")

# Custom CSS for the "Deepak Chopra" Aesthetic (Light, Minimal, Elegant)
st.markdown("""
    <style>
    /* Background and overall font */
    .stApp { 
        background-color: #fdfcfb; 
        color: #333333;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Center the title and make it elegant */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 300;
        color: #2d3436;
        margin-top: 50px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #636e72;
        margin-bottom: 40px;
    }

    /* Minimalist Search Bar */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #333333 !important; 
        border: 1px solid #e0e0e0 !important;
        border-radius: 30px !important; /* Rounded like Deepak's site */
        padding: 15px 25px !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
    }

    /* Floating Wisdom Card */
    .wisdom-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 40px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        margin-top: 30px;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2d3436;
        animation: fadeIn 0.8s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hide Streamlit default elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. AI Engine Setup
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

# 3. Main UI Layout
st.markdown('<h1 class="main-title">Maya-GPT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A Bridge Between Science, Philosophy, and Consciousness</p>', unsafe_allow_html=True)

# Central Input
query = st.text_input("", placeholder="Ask a question or share a thought...", label_visibility="collapsed")

if query:
    with st.spinner("Reflecting..."):
        # RAG Logic
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_prompt = (
            "You are Maya-GPT, inspired by the depth of Deepak Chopra and the precision of Quantum Physics. "
            "Your purpose is to provide reflections based on Vedanta, Western Philosophy, and Science. "
            "Be empathetic, wise, and clear. You can answer in any language the user asks in.\n\n"
            f"Context: {context}\n\nQuestion: {query}"
        )
        
        response = llm.invoke(system_prompt)
        
        # Display response in the "Zen" card
        st.markdown(f'<div class="wisdom-card">{response.content}</div>', unsafe_allow_html=True)

st.markdown("<br><br><p style='text-align:center; color:#b2bec3; font-size:0.8rem;'>Inspired by Universal Wisdom & Modern Science</p>", unsafe_allow_html=True)
