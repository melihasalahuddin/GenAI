import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="RAG-based Assistant", layout="wide")
st.title("RAG Application using Gemini Pro")
st.subheader("Ask questions about the uploaded document")

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Process the uploaded file
if uploaded_file is not None:
    # Display processing message
    with st.spinner("Processing document..."):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and process the PDF
        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        
        # Create embeddings and FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        st.success(f"Document processed: {uploaded_file.name}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Configure LLM and chain
def create_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None
    )
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use ten sentences maximum and keep the "
        "answer concise. Only extract content from the pdf."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Get user input
query = st.chat_input("Ask me anything about the uploaded document", disabled=st.session_state.vectorstore is None)

# Process the query
if query and st.session_state.vectorstore is not None:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create chain and get response
            rag_chain = create_chain(st.session_state.vectorstore)
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
            st.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display a message if no document is uploaded
if st.session_state.vectorstore is None:
    st.info("Please upload a pdf document to start asking questions.")


