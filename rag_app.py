import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="RAG Multi-PDF 💬", layout="wide")

st.title("Conversational PDF Assistant")

st.write("Upload up to **10 PDF files** and ask questions about their content.")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://unsplash.com/fr/photos/jouet-robot-en-acier-inoxydable-en-photographie-en-gros-plan-_3KdlCgHAn0");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- UPLOAD PDF FILES ---
uploaded_files = st.file_uploader("Téléverse tes fichiers PDF :", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    # --- CHARGEMENT PDF ---
    for uploaded_file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)

    st.success(f"{len(uploaded_files)} Files uploaded successfully ✅")

    # --- CHUNKING ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # --- EMBEDDINGS + VECTORSTORE ---
    st.write("Creating embeddings... (Please, wait !)")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # --- LLM + RAG CHAIN ---
    llm = ChatOpenAI(model="gpt-4o-mini")  # ou "gpt-4-turbo"
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # --- CHAT INPUT ---
    st.divider()
    user_query = st.text_input("Ask your question here :")

    if user_query:
        with st.spinner("Searching in PDFs..."):
            result = qa_chain.invoke({"query": user_query})

        st.subheader("Réponse :")
        st.write(result["result"])

        with st.expander("Sources used"):
            for doc in result["source_documents"]:
                st.markdown(f"- **{doc.metadata['source']}**")

else:
    st.info("Upload PDFs to get started.")
