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
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

user_query = st.text_input("Ask Maya-GPT:")

if user_query:
    # 1. Increase search depth to 'k=5' to get more of your data
    docs = retriever.invoke(user_query)
    
    # 2. Combine the retrieved snippets into one 'context' block
    context = "\n\n".join([d.page_content for d in docs])
    
    # 3. Use the 'Stronger Prompt' to keep it focused on your 701 chunks
    system_prompt = (
        "You are Maya-GPT, an expert in the intersection of Vedanta and Science. "
        "Use ONLY the following pieces of retrieved context to answer the question. "
        "Stay grounded in the provided text. If the answer isn't in the context, "
        "honestly say it isn't explicitly mentioned, but offer a perspective "
        "based on the available data.\n\n"
        f"Context: {context}\n\n"
        f"Question: {user_query}"
    )
    
    # 4. Get the answer from the AI
    response = llm.invoke(system_prompt)
    st.write(response.content)

    # 5. Optional: See what 'chunks' were used (for your own verification)
    with st.expander("🔍 View the Source Chunks used for this answer"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**\n{doc.page_content[:200]}...")
