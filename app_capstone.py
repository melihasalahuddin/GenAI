import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import tempfile
import random
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# Handle various import scenarios for langchain-groq
try:
    from langchain_groq import ChatGroq
except ImportError:
    try:
        # Try alternative import path
        from langchain.chat_models import ChatGroq
    except ImportError:
        # If all else fails, use a generic model
        from langchain_community.chat_models import ChatOpenAI as ChatGroq
        print("WARNING: Could not import ChatGroq. Using ChatOpenAI as a fallback.")
from datetime import datetime

# Load environment variables from .env file if present
load_dotenv()

# Initialize session state for persistence across reruns
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'pdf_files_uploaded' not in st.session_state:
    st.session_state.pdf_files_uploaded = False
if 'csv_uploaded' not in st.session_state:
    st.session_state.csv_uploaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'advance_summary' not in st.session_state:
    st.session_state.advance_summary = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App title and description
st.title("Business Intelligence Assistant")
st.markdown("""
This app analyzes structured (CSV) and unstructured (PDF) data to generate insights and recommendations.
Upload your business documents and sales data to get started.
""")

# Function to initialize Groq LLM
def init_groq_llm(model_name="llama-3.1-70b-versatile", temperature=0.3, max_tokens=2048):
    llm = ChatGroq(
        model=model_name,
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens
    )
    # Store model name as an attribute for later reference
    llm._model_name = model_name
    return llm


# Sidebar for API key and settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter your Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    model_name = st.selectbox("Select model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama-3.1-70b-instant", "llama-3.1-8b-versatile"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)  # Increased default temperature for more varied responses
    max_tokens = st.slider("Max tokens", 100, 4096, 2048, 100)
    
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        # Check if we need to initialize or update the LLM
        if st.session_state.llm is None or getattr(st.session_state.llm, '_model_name', '') != model_name:
            st.session_state.llm = init_groq_llm(model_name, temperature, max_tokens)
            st.success(f"Connected to Groq with model: {model_name}")



# Function to load and process PDF files
def load_pdf_files(uploaded_files):
    doc = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary directory
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Load PDF file
            loader = PyPDFLoader(temp_file_path)
            doc.extend(loader.load())
    
    return doc

# Function to split documents into chunks
def split_documents(documents, chunk_size=100, chunk_overlap=20):
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(documents)
    return texts

# Function to set up vector database and retriever
def setup_vector_database(texts):
    # Initialize Hugging Face embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Set up retriever with slightly higher k for more diverse results
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    return retriever

# Function to generate advanced data summary from sales data
def generate_advanced_data_summary(df):
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sales Analysis
    total_sales = df['Sales'].sum()
    avg_sale = df['Sales'].mean()
    median_sale = df['Sales'].median()
    sales_std = df['Sales'].std()

    # Time-based Analysis
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month', observed=False)['Sales'].sum().sort_values(ascending=False)
    best_month = monthly_sales.index[0]
    worst_month = monthly_sales.index[-1]

    # Product Analysis
    product_sales = df.groupby('Product', observed=False)['Sales'].agg(['sum', 'count', 'mean'])
    top_product = product_sales['sum'].idxmax()
    most_sold_product = product_sales['count'].idxmax()

    # Regional Analysis
    region_sales = df.groupby('Region', observed=False)['Sales'].sum().sort_values(ascending=False)
    best_region = region_sales.index[0]
    worst_region = region_sales.index[-1]

    # Customer Analysis
    avg_satisfaction = df['Customer_Satisfaction'].mean()
    satisfaction_std = df['Customer_Satisfaction'].std()

    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, labels=age_labels, right=False)
    age_group_sales = df.groupby('Age_Group', observed=False)['Sales'].mean().sort_values(ascending=False)
    best_age_group = age_group_sales.index[0]

    # Gender Analysis
    gender_sales = df.groupby('Customer_Gender', observed=False)['Sales'].mean()

    summary = f"""
    Advanced Sales Data Summary:

    Overall Sales Metrics:
    - Total Sales: ${total_sales:,.2f}
    - Average Sale: ${avg_sale:.2f}
    - Median Sale: ${median_sale:.2f}
    - Sales Standard Deviation: ${sales_std:.2f}

    Time-based Analysis:
    - Best Performing Month: {best_month}
    - Worst Performing Month: {worst_month}

    Product Analysis:
    - Top Selling Product (by value): {top_product}
    - Most Frequently Sold Product: {most_sold_product}

    Regional Performance:
    - Best Performing Region: {best_region}
    - Worst Performing Region: {worst_region}

    Customer Insights:
    - Average Customer Satisfaction: {avg_satisfaction:.2f}/5
    - Customer Satisfaction Standard Deviation: {satisfaction_std:.2f}
    - Best Performing Age Group: {best_age_group}
    - Gender-based Average Sales: Male=${gender_sales.get('Male', 0):.2f}, Female=${gender_sales.get('Female', 0):.2f}

    Key Observations:
    1. The sales data shows significant variability with a standard deviation of ${sales_std:.2f}.
    2. The {best_age_group} age group shows the highest average sales.
    3. Regional performance varies significantly, with {best_region} outperforming {worst_region}.
    4. The most valuable product ({top_product}) differs from the most frequently sold product ({most_sold_product}), suggesting potential for targeted marketing strategies.
    """

    return summary

# Function to create a prompt template for sales data analysis
def create_sales_prompt_template():
    scenario_template = """
    You are an expert AI Sales Analyst. Use the following advance data summary to provide answers to the question at the end and give some actionable recommendation
    Be Specific and refer to the data points provided.

    {advance_summary}

    Question: {question}

    Detailed Analysis and Recommendation
    """

    return PromptTemplate(template=scenario_template, input_variables=["advance_summary", "question"])

# Function to create extraction prompt for RAG
def create_extraction_prompt():
    extraction_template = """
    You are an expert data analyst tasked with extracting and organizing key information from retrieved documents.

    Please analyze the following retrieved documents carefully and extract:
    1. Key facts, data points, and statistics
    2. Main themes and concepts
    3. Potential insights or implications
    4. Any inconsistencies or gaps in the information

    Format your response as a structured analysis that can be used by a strategic advisor in the next stage.

    Retrieved Documents:
    {context}

    Query: {query}

    Structured Analysis:
    """

    return PromptTemplate(template=extraction_template, input_variables=["context", "query"])

# Function to create synthesis prompt for RAG with conversation history
def create_synthesis_prompt():
    synthesis_template = """
    You are a strategic business advisor providing expert guidance based on thorough analysis.

    Review the structured analysis below, which was extracted from relevant documents regarding the query.

    Structured Analysis:
    {extracted_analysis}

    Additional Context:
    {advance_summary}

    Previous Conversation:
    {conversation_history}

    Current Query: {query}

    Based on this information, provide:
    1. A comprehensive answer to the query
    2. Strategic recommendations with clear rationale
    3. Potential implementation steps
    4. Anticipated outcomes and metrics for success

    Your response should be clear, actionable, and directly address the query with specific references to the data.
    Vary your style and approach to avoid repetitive responses - focus on new insights or different perspectives if the user asks similar questions.
    """

    return PromptTemplate(template=synthesis_template, input_variables=["extracted_analysis", "advance_summary", "conversation_history", "query"])

# Function to format conversation history
def format_conversation_history(chat_history, max_entries=3):
    if not chat_history or len(chat_history) == 0:
        return "No previous conversation."
    
    # Take last few entries to provide recent context
    recent_history = chat_history[-min(max_entries*2, len(chat_history)):]
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
    return formatted_history

# Function to add slight variation to the query for more diverse retrieval
def add_query_variation(query):
    # List of query enhancers that don't change the meaning but add variety
    enhancers = [
        "", 
        "I'd like information about ",
        "Can you tell me about ",
        "I'm interested in ",
        "Let's explore ",
        "Please analyze "
    ]
    
    # Randomly select an enhancer
    enhancer = random.choice(enhancers)
    
    # If enhancer is empty, return original query
    if enhancer == "":
        return query
    
    # Otherwise, add enhancer to query
    return f"{enhancer}{query}"

# Function to set up two-stage QA chain with conversation history
def setup_two_stage_qa_chain(llm, retriever, advance_summary):
    # Use the existing LLM instance for both stages
    extraction_llm = llm  # Use the same LLM for extraction
    synthesis_llm = llm   # Use the same LLM for synthesis

    # Create prompts
    extraction_prompt = create_extraction_prompt()
    synthesis_prompt = create_synthesis_prompt()

    # Document retriever wrapper function with chat history
    def retrieve_and_extract(query, chat_history=None):
        # Format conversation history for context
        conversation_context = format_conversation_history(chat_history) if chat_history else "No previous conversation."
        
        # Add slight variation to query for document retrieval to get different results
        varied_query = add_query_variation(query)
        
        # Get documents from retriever
        docs = retriever.get_relevant_documents(varied_query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Extract structured analysis
        extraction_chain = LLMChain(prompt=extraction_prompt, llm=extraction_llm)
        extracted_analysis = extraction_chain.run(context=context, query=query)

        # Generate final synthesized response with conversation history
        synthesis_chain = LLMChain(prompt=synthesis_prompt, llm=synthesis_llm)
        final_response = synthesis_chain.run(
            extracted_analysis=extracted_analysis,
            advance_summary=advance_summary if advance_summary else "No sales data summary available.",
            conversation_history=conversation_context,
            query=query
        )

        return {
            "result": final_response,
            "source_documents": docs,
            "extracted_analysis": extracted_analysis
        }

    return retrieve_and_extract

# Function to generate insights from sales data
def generate_insights(llm, prompt, advance_summary, question):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(advance_summary=advance_summary, question=question)

# Tabs for data upload, analysis, and chat
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Data Preview", "Data Analysis", "Chat"])

# Tab 1: Data Upload
with tab1:
    st.header("Upload Data")
    
    # PDF Upload
    pdf_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type="pdf")
    if pdf_files:
        st.session_state.pdf_files = pdf_files
        if st.button("Process PDF Files"):
            with st.spinner("Processing PDF files..."):
                doc = load_pdf_files(pdf_files)
                st.write(f"Loaded {len(doc)} PDF pages")
                
                texts = split_documents(doc)
                st.write(f"Created {len(texts)} text chunks")
                
                st.session_state.retriever = setup_vector_database(texts)
                st.session_state.pdf_files_uploaded = True
                st.success("PDF files processed successfully!")
    
    # CSV Upload
    csv_file = st.file_uploader("Upload sales data (CSV)", type="csv")
    if csv_file:
        with st.spinner("Processing CSV file..."):
            df = pd.read_csv(csv_file)
            st.session_state.df = df
            st.session_state.csv_uploaded = True
            
            # Generate advanced summary
            try:
                st.session_state.advance_summary = generate_advanced_data_summary(df)
                st.success("Sales data processed successfully!")
            except Exception as e:
                st.error(f"Error processing sales data: {str(e)}")

# Tab 2: Data Preview
with tab2:
    st.header("Data Preview")
    
    # Preview CSV data
    if st.session_state.csv_uploaded:
        st.subheader("Sales Data")
        st.dataframe(st.session_state.df)
    else:
        st.info("Please upload a CSV file in the Data Upload tab.")
    
    # Show PDF processing status
    if st.session_state.pdf_files_uploaded:
        st.subheader("PDF Documents")
        st.success("PDF documents processed and ready for querying.")
    else:
        st.info("Please upload and process PDF files in the Data Upload tab.")

# Tab 3: Data Analysis
with tab3:
    st.header("Data Analysis")
    
    if st.session_state.csv_uploaded:
        st.subheader("Sales Data Summary")
        st.text(st.session_state.advance_summary)
        
        st.subheader("Generate Sales Insights")
        question = st.text_input("Ask a question about the sales data:")
        
        if question and st.session_state.llm:
            if st.button("Generate Insights"):
                with st.spinner("Generating insights..."):
                    prompt = create_sales_prompt_template()
                    insights = generate_insights(st.session_state.llm, prompt, st.session_state.advance_summary, question)
                    st.markdown("### Insights")
                    st.markdown(insights)
    else:
        st.info("Please upload sales data (CSV) in the Data Upload tab to see analysis.")

# Tab 4: Chat Interface
with tab4:
    st.header("Chat with your Documents")
    
    if not (st.session_state.pdf_files_uploaded or st.session_state.csv_uploaded):
        st.info("Please upload and process PDF files and/or CSV data first.")
    elif st.session_state.llm is None:
        st.info("Please enter your Groq API Key in the sidebar.")
    else:
        # Setup two-stage QA chain if not already set up
        if 'qa_chain' not in st.session_state and st.session_state.retriever:
            st.session_state.qa_chain = setup_two_stage_qa_chain(
                st.session_state.llm,
                st.session_state.retriever,
                st.session_state.advance_summary if st.session_state.csv_uploaded else None
            )
        
        # Add reset button for conversation
        if st.button("Reset Conversation"):
            st.session_state.chat_history = []
            st.success("Conversation has been reset!")
            st.rerun()
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
        
        # Chat input
        user_query = st.text_input("Ask a question about your documents:")
        
        if user_query and st.session_state.qa_chain:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            with st.spinner("Generating response..."):
                # Pass chat history to the QA chain
                result = st.session_state.qa_chain(user_query, st.session_state.chat_history)
                response = result["result"]
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Force a rerun to display the new messages
                st.rerun()

# Display information about what this app can do
with st.sidebar:
    st.markdown("---")
    st.subheader("About this app")
    st.markdown("""
    This Business Intelligence Assistant can:
    - Process PDF documents and sales data
    - Generate advanced data summaries
    - Answer questions about your business data
    - Provide strategic recommendations
    - Analyze trends and patterns in your data
    
    For best results, upload both PDF documents and sales data.
    """)