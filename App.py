import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from datetime import datetime
import logging
from collections import deque
import os

# Improved logging configuration
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@st.cache_data
def load_embedding_model(model_path, normalize_embedding=True):
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding
        }
    )

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text_pages = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    text_pages.append((text, i + 1, pdf.name))
        except Exception as e:
            logging.error(f"Error reading PDF file {pdf.name}: {e}")
            st.error(f"Error reading PDF file {pdf.name}. Check logs for details.")
    return text_pages

def get_text_chunks(text_pages):
    """Split text into chunks for processing."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = []
        for text, page_num, pdf_name in text_pages:
            split_chunks = text_splitter.split_text(text)
            chunks.extend((chunk, page_num, pdf_name) for chunk in split_chunks)
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        st.error("Error splitting text into chunks. Check logs for details.")
        return []

def get_vector_store(text_chunks):
    """Create and save vector store from text chunks."""
    try:
        embeddings = load_embedding_model(model_path="all-MiniLM-L6-v2")
        texts = [chunk for chunk, _, _ in text_chunks]
        metadatas = [{"page_num": page_num, "pdf_name": pdf_name} for _, page_num, pdf_name in text_chunks]
        vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error("Error creating vector store. Check logs for details.")

def get_conversational_chain():
    """Set up the conversational chain for question answering."""
    try:
        prompt_template = """
        You are a respectful and honest assistant. Answer the user's 
        questions using only the context provided to you.
        Explain in a way that ensures the person understands the concept clearly.
        History:\n{memory}
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer in as much detail as possible:
        """
        llm = Ollama(model="llama3", temperature=0)
        prompt = PromptTemplate(template=prompt_template, input_variables=["memory", "context", "question"])
        return load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    except Exception as e:
        logging.error(f"Error setting up conversational chain: {e}")
        st.error("Error setting up conversational chain. Check logs for details.")
        return None

def user_input(user_question, memory):
    """Process user input and generate a response."""
    try:
        embeddings = load_embedding_model(model_path="all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain:
            response = chain({"memory": memory, "input_documents": docs, "question": user_question}, return_only_outputs=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.write("Reply: ", response["output_text"])
            citations = [
                f"PDF `{doc.metadata['pdf_name']}`, Page `{doc.metadata['page_num']}`: {doc.page_content[:100]} ... {doc.page_content[-50:]}"
                for doc in docs
            ]
            if citations:
                st.write("CITATIONS (for the above response)")
                for citation in citations:
                    st.write(citation)
                st.write("---------")
            return timestamp, response['output_text']
        else:
            st.error("Conversational chain setup failed. Check logs for details.")
            return None, None
    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        st.error("Error processing user input. Check logs for details.")
        return None, None

@st.cache_data
def read_last_lines(filename, lines_count):
    """Read the last n lines from a file."""
    try:
        with open(filename, 'r') as file:
            return ''.join(deque(file, maxlen=lines_count))
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
        return "Error reading log file. Check logs for details."

def main():
    st.set_page_config(page_title="DOCHAT")
    st.header("DOCHAT")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Main chat interface
    user_question = st.text_input("Ask a question about the uploaded PDFs")

    if user_question:
        timestamp, ai_response = user_input(user_question, st.session_state['chat_history'])
        if timestamp and ai_response:
            st.session_state['chat_history'].append(("----------\nTime", timestamp))
            st.session_state['chat_history'].append(("USER", user_question))
            st.session_state['chat_history'].append(("AI", ai_response))

    # Sidebar
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                text_pages = get_pdf_text(pdf_docs)
                if text_pages:
                    text_chunks = get_text_chunks(text_pages)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("Failed to chunk text. Check logs for details.")
                else:
                    st.error("Failed to process PDF files. Check logs for details.")

        if st.button("Clear Chat History"):
            st.session_state['chat_history'] = []
            st.success("Chat history cleared!")

        show_logs = st.checkbox("Show Logs")
        if show_logs:
            st.subheader("Recent Logs")
            last_lines = read_last_lines("app.log", 10)
            st.text(last_lines)

    # Display chat history
    if st.session_state['chat_history']:
        st.subheader("Chat History")
        for role, text in st.session_state['chat_history']:
            st.text(f"{role}: {text}")
        st.markdown("---")

if __name__ == "__main__":
    main()