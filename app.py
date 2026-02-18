import streamlit as st
import os

# 1. THE SECURE WAY: Get the key from the environment/secrets
# When you deploy, we will put this in the Streamlit Dashboard settings
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="Maya-GPT", page_icon="🧘")
st.title("🧘 Maya-GPT: The Bridge")

if not api_key:
    st.error("API Key not found. Please set GROQ_API_KEY in Secrets.")
    st.stop()

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

class SimpleEmbedder:
    def __init__(self): self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_documents(self, texts): return self.model.encode(texts).tolist()
    def embed_query(self, text): return self.model.encode([text])[0].tolist()

# The rest of your working logic...
embeddings = SimpleEmbedder()
vectorstore = Chroma(persist_directory="./maya_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

user_query = st.text_input("Ask Maya-GPT:")
if user_query:
    docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([d.page_content for d in docs])
    response = llm.invoke(f"Context: {context}\n\nQuestion: {user_query}")
    st.write(response.content)
