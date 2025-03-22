import streamlit as st
import os
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set page config
st.set_page_config(
    page_title="RAG with Gemini",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("📚 RAG with Google Gemini")
st.subheader("Upload a PDF and ask questions about it")
st.markdown("This app uses Google's Gemini model to answer questions about your documents.")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API Key:", type="password")
    
    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        "This app demonstrates Retrieval-Augmented Generation (RAG) using:\n"
        "- Google Gemini embeddings\n"
        "- Chroma vector store\n"
        "- Document chunking\n"
        "- Google Gemini LLM"
    )

# Function to process PDF
def process_pdf(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        return docs, f"✅ Processed PDF: {len(docs)} chunks created"
    except Exception as e:
        return None, f"❌ Error processing PDF: {str(e)}"
    finally:
        # Clean up temp file
        os.unlink(pdf_path)

# Additional helper function to add a document summary feature
def generate_document_summary(rag_chain, docs):
    try:
        # Create a summary prompt
        summary_question = "Please provide a comprehensive summary of this document. What are the main topics, findings, and conclusions?"
        response = rag_chain.invoke({"input": summary_question})
        return response["answer"], None
    except Exception as e:
        return None, f"Error generating summary: {str(e)}"

# Function to set up RAG pipeline
def setup_rag(docs, api_key):
    try:
        # Set API key
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Create embeddings and vector store with persistent directory
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Use an in-memory Chroma instance
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings,
            collection_name="document_collection",
            persist_directory=None  # In-memory only, no persistence
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        # Set up LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)
        
        # Create RAG chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use the retrieved context to provide a comprehensive "
            "and accurate answer."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain, "✅ RAG pipeline setup complete"
    except Exception as e:
        return None, f"❌ Error setting up RAG: {str(e)}"

# Main app flow
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Initialize session state
if "docs" not in st.session_state:
    st.session_state.docs = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None

# Process uploaded PDF
if uploaded_file is not None and not st.session_state.processing_complete:
    with st.spinner("Processing PDF..."):
        docs, message = process_pdf(uploaded_file)
        st.session_state.docs = docs
        st.info(message)
        
        if docs and api_key:
            with st.spinner("Setting up RAG pipeline..."):
                rag_chain, setup_message = setup_rag(docs, api_key)
                st.session_state.rag_chain = rag_chain
                st.info(setup_message)
                if rag_chain:
                    st.session_state.processing_complete = True
                    
                    # Generate document summary automatically
                    with st.spinner("Generating document summary..."):
                        summary, summary_error = generate_document_summary(rag_chain, docs)
                        if summary:
                            st.session_state.document_summary = summary
                        else:
                            st.session_state.document_summary = "Unable to generate summary."
                            if summary_error:
                                st.warning(summary_error)
                    
                    st.success("Ready to answer questions about your document!")
        elif not api_key:
            st.warning("Please enter your Google API Key in the sidebar")

# Reset app button
if st.session_state.processing_complete:
    if st.button("Process a new document"):
        st.session_state.docs = None
        st.session_state.rag_chain = None
        st.session_state.document_summary = None
        st.session_state.processing_complete = False
        st.experimental_rerun()

# Question answering section
if st.session_state.processing_complete and st.session_state.rag_chain:
    st.markdown("---")
    
    # Display document summary
    if st.session_state.document_summary:
        st.subheader("Document Summary")
        st.write(st.session_state.document_summary)
    
    st.markdown("---")
    st.subheader("Ask questions about your document")
    
    # Predefined questions as quick selection
    predefined_questions = [
        "What is the main topic of this document?",
        "Summarize the key findings",
        "What are the conclusions?",
        "Explain the methodology used",
        "What are the limitations mentioned?",
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Enter your question:")
    with col2:
        selected_question = st.selectbox(
            "Or select a question:",
            ["Custom question"] + predefined_questions,
            index=0
        )
    
    # Set the question based on selection
    if selected_question != "Custom question":
        question = selected_question
    
    if question:
        with st.spinner("Generating answer..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": question})
                
                # Display response
                st.markdown("### Answer")
                st.write(response["answer"])
                
                # Display sources (optional expandable section)
                with st.expander("View source documents"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source {i+1}**")
                        st.markdown(f"```\n{doc.page_content}\n```")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.info("Try uploading the document again or check your API key.")