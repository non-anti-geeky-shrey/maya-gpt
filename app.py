import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Security & Configuration
api_key = st.secrets.get("GROQ_API_KEY")

st.set_page_config(page_title="Maya-GPT", page_icon="🧘")
st.title("🧘 Maya-GPT: The Universal Bridge")
st.markdown("### Bridging Quantum Physics, Consciousness, and Eastern Vedantic Philosophy")

if not api_key:
    st.error("API Key not found in Secrets! Please add GROQ_API_KEY to your Streamlit settings.")
    st.stop()

# 2. Setup the "Memory" (Vector DB)
# Using the MiniLM model you trained in Colab
class SimpleEmbedder:
    def __init__(self): 
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, texts): 
        return self.model.encode(texts).tolist()
    def embed_query(self, text): 
        return self.model.encode([text])[0].tolist()

@st.cache_resource
def load_vectorstore():
    embeddings = SimpleEmbedder()
    return Chroma(persist_directory="./maya_db", embedding_function=embeddings)

vectorstore = load_vectorstore()
# k=4 provides a balanced depth of retrieval from your 701 chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

# 3. User Interface
user_query = st.text_input("Ask about Physics, Philosophy, or Consciousness:")

if user_query:
    with st.spinner("Synthesizing across dimensions..."):
        # Search the knowledge base
        docs = retriever.invoke(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        
        # The Balanced Universal Bridge Prompt
        system_prompt = (
            "You are Maya-GPT. Your purpose is to bridge the gap between "
            "Quantum Physics, Consciousness, Eastern Philosophy (Vedanta), and Western Philosophy. "
            "\n\n--- CONTEXT FROM YOUR DATABASE ---\n"
            f"{context}"
            "\n\n--- YOUR GUIDELINES ---\n"
            "1. Synthesize insights from the provided context with your deep knowledge of physics and philosophy.\n"
            "2. Do not say 'This is not explicitly defined.' Jump straight into a wise, analytical, and poetic explanation.\n"
            "3. Show how these different fields might be describing the same reality from different angles (e.g., relating Non-duality to Quantum Entanglement).\n"
            "4. Speak with the authority of a sage and the precision of a scientist.\n\n"
            f"Question: {user_query}"
        )
        
        # Generate and display response
        response = llm.invoke(system_prompt)
        st.markdown(f"**Maya's Insight:**\n\n{response.content}")

        # Verification tool for you to check your data
        with st.expander("🔍 View the Source Chunks Used"):
            for i, doc in enumerate(docs):
                st.info(f"**Chunk {i+1}:**\n{doc.page_content}")
