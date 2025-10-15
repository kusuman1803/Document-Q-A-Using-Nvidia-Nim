import streamlit as st
import os
import time
import tempfile
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# -------------------------------
# Streamlit UI Configuration
# -------------------------------
st.set_page_config(page_title="Nvidia NIM Demo", page_icon="🧠")

st.title("🤖 Nvidia NIM Demo with FAISS & LangChain")

# Sidebar for API Key input
st.sidebar.header("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your NVIDIA API Key:", type="password")

if not api_key:
    st.warning("⚠️ Please enter your NVIDIA API Key in the sidebar to continue.")
    st.stop()

# Set environment variable from input
os.environ["NVIDIA_API_KEY"] = api_key

# -------------------------------
# File Upload
# -------------------------------
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------
# LLM and Embedding Setup
# -------------------------------
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embedding(uploaded_files):
    """Generate vector embeddings from uploaded PDFs and store in FAISS"""
    if not uploaded_files:
        st.error("❌ Please upload at least one PDF file.")
        return

    if "vectors" not in st.session_state:
        with st.spinner("📄 Processing and embedding uploaded PDFs..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Load PDF content
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            final_docs = text_splitter.split_documents(all_docs)

            # Create embeddings
            embeddings = NVIDIAEmbeddings()
            st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
            st.session_state.docs_ready = True

        st.success("✅ FAISS Vector Store is ready using NVIDIA Embeddings!")

# -------------------------------
# Prompt Template
# -------------------------------
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
Question: {input}
"""
)

# -------------------------------
# User Question
# -------------------------------
prompt1 = st.text_input("💬 Ask your question based on uploaded PDFs:")

if st.button("📚 Create Document Embeddings"):
    vector_embedding(uploaded_files)

if prompt1:
    if "vectors" not in st.session_state:
        st.error("❌ Please upload PDFs and create document embeddings first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed_time = time.process_time() - start

        st.success(f"✅ Response generated in {elapsed_time:.2f} seconds")
        st.markdown("### 🧠 Answer:")
        st.write(response["answer"])

        with st.expander("📎 Document Similarity Search Context"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**🔹 Document {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
