import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the persistent directory for Chroma
persist_directory = './data/chroma/'

# Initialize the OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Initialize the Chroma vector store with the persistent directory
vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# Initialize session state keys if they aren't already present
if 'temp_dir' not in st.session_state:
    st.session_state['temp_dir'] = tempfile.mkdtemp()

# Function to load documents
def load_document(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            st.error("Unsupported file type.")
            return None
        docs = loader.load()
        return docs
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None

# Function to split documents
def split_documents(docs):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        return all_splits
    except Exception as e:
        st.error(f"Error during splitting documents: {e}")
        return None

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to chat with the model
def chat_with_model(input, retriever):
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.invoke(input)
    except Exception as e:
        st.error(f"Error during chat with model: {e}")
        return "Error during chat."

# Streamlit UI
st.title("Document-based Chatbot")
st.write("Upload PDF or CSV files and chat with the bot about their contents.")

uploaded_files = st.file_uploader("Choose files", type=["pdf", "csv"], accept_multiple_files=True)

if uploaded_files:
    process_files = st.button("Process Files")
    if process_files:
        for uploaded_file in uploaded_files:
            # Save the uploaded file
            file_path = os.path.join(st.session_state['temp_dir'], uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Load, process, and store the information about the uploaded file
            docs = load_document(file_path)
            if docs:
                splits = split_documents(docs)
                if splits:
                    vectordb.add_documents(documents=splits)
                    st.success(f"File {uploaded_file.name} processed and added to the database.")
                else:
                    st.error(f"Failed to split document: {uploaded_file.name}")
            else:
                st.error(f"Failed to load document: {uploaded_file.name}")

# Querying phase
retriever = vectordb.as_retriever()
if retriever:
    user_input = st.text_input("Ask a question about the documents:")
    if user_input:
        answer = chat_with_model(user_input, retriever)
        st.write(answer)
else:
    st.info("Please upload some files and click 'Process Files' to proceed.")
