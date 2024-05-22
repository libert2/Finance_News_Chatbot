import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_paths): 
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_or_update_vector_store(pdf_paths):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index"  

    try:
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        vector_store = None

    new_pdfs = []
    for path in pdf_paths:
        if path not in vector_store.docstore._dict.keys():  # Check if PDF is already indexed
            new_pdfs.append(path)

    if new_pdfs:
        new_text = get_pdf_text(new_pdfs)
        new_chunks = get_text_chunks(new_text)

        if vector_store:
            vector_store.add_texts(new_chunks) 
        else:
            vector_store = FAISS.from_texts(new_chunks, embeddings)

        vector_store.save_local(index_path)  
    return vector_store

def get_conversational_chain():
    # (Your existing get_conversational_chain implementation remains unchanged)
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.7)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    if response is not None and "output_text" in response:
        print(response)
        st.write("Reply: ", response["output_text"])
        return response
    else:
        st.write("Reply: No relevant information found in the documents.")
        return None

def main():
    st.set_page_config("Chat PDF")
    st.header("Finance News Chatbot")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    with st.sidebar:
        st.subheader("Chat History")
        for entry in st.session_state.chat_history:
            with st.expander(entry["question"][:50] + "..."):  
                st.write("**Question:**", entry["question"])
                st.write("**Answer:**", entry["answer"])
    # Replace with paths to your local PDFs (ensure they exist)
    default_pdf_paths = [
        "D:\\LLM Krish Naik\\ChatWithMultiplePdf\\output_file.pdf",
        "D:\\LLM Krish Naik\\ChatWithMultiplePdf\\output_file_2018.pdf",
        "D:\\LLM Krish Naik\\ChatWithMultiplePdf\\output_file_2019.pdf"
        # Add more PDF paths as needed
    ]
    
    # Display chat history in the sidebar first 
    

    user_question = st.text_input("Ask a Question")
    
    # Load or update the vector store only once
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_or_update_vector_store(default_pdf_paths)
        
    if user_question:
        response = user_input(user_question, st.session_state.vector_store)
        if response is not None:
            st.session_state.chat_history.append(
                {"question": user_question, "answer": response["output_text"]}
            )
    
if __name__ == "__main__":
    main()
    # Display chat history in the sidebar
    # (Your existing chat history display code remains unchanged)
