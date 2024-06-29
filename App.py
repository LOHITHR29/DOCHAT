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

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_pdf_text(pdf_docs):
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

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding
        }
    )

def get_text_chunks(text_pages):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = []
        for text, page_num, pdf_name in text_pages:
            split_chunks = text_splitter.split_text(text)
            for chunk in split_chunks:
                chunks.append((chunk, page_num, pdf_name))
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        st.error("Error splitting text into chunks. Check logs for details.")
        return []

def get_vector_store(text_chunks):
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
    try:
        prompt_template = """
        You are an respectful and honest assistant. You have to answer the user's 
        questions using only the context provided to you.
        You have to explain in a way so that the person understands well the concept and is clear regarding the same.
        History:\n{memory}
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer in as detail as possible

        Answer:
        """
        llm = Ollama(model="llama3", temperature=0)
        prompt = PromptTemplate(template=prompt_template, input_variables=["memory", "context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error setting up conversational chain: {e}")
        st.error("Error setting up conversational chain. Check logs for details.")
        return None

def user_input(user_question, memory):
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
                f"PDF `{doc.metadata['pdf_name']}`, Page `{doc.metadata['page_num']}`: {doc.page_content[:100]} . . . . . . .{doc.page_content[-50:]}"
                for doc in docs
            ]
            if citations:
                st.write("`CITATIONS (for the above response)`")
                for role in citations:
                    st.write(role)
                st.write("---------")
            return timestamp, response['output_text']
        else:
            st.error("Conversational chain setup failed. Check logs for details.")
            return None, None
    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        st.error("Error processing user input. Check logs for details.")
        return None, None

def read_last_lines(filename, lines_count):
    with open(filename, 'r') as file:
        return ''.join(deque(file, maxlen=lines_count))

def main():
    st.set_page_config(page_title="Alameno Bot")
    st.header("ALAN BOT")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        timestamp, ai_response = user_input(user_question, st.session_state['chat_history'])
        if timestamp and ai_response:
            st.session_state['chat_history'].append(("----------\n`Time`", timestamp))
            st.session_state['chat_history'].append(("`USER`", user_question))
            st.session_state['chat_history'].append(("`AI`", ai_response))

    with st.sidebar:
        st.title("Menu:")
        on = st
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                text_pages = get_pdf_text(pdf_docs)
                if text_pages:
                    text_chunks = get_text_chunks(text_pages)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("Done")
                else:
                    st.error("Failed to process PDF files. Check logs for details.")

        st.write("*`by Ayon Somaddar`*")

    if st.session_state['chat_history']:
        st.title("Chat History")
        for role, text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")
        st.write("-----")

    with st.sidebar:
        on = st.checkbox("log")
        if on:
            st.title("Logs")
            last_lines = read_last_lines("app.log", 5)
            st.text(last_lines)

if __name__ == "__main__":
    main()
