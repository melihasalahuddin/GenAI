import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import wikipedia
from bs4 import BeautifulSoup

# Langchain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, LLMChain
from langchain.tools import Tool
from langchain.chains import RetrievalQA, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, ZeroShotAgent
from langchain_groq import ChatGroq

# Set page config
st.set_page_config(
    page_title="Sales Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Set API key
os.environ["GROQ_API_KEY"] = "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"

# Create sidebar for navigation
st.sidebar.title("Sales Analysis Dashboard")
page = st.sidebar.radio("Navigation", ["Home", "Data Upload", "Data Analysis", "RAG System", "Agent Analysis"])

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'advanced_summary' not in st.session_state:
    st.session_state.advanced_summary = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'sales_summary_doc' not in st.session_state:
    st.session_state.sales_summary_doc = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'pdf_docs' not in st.session_state:
    st.session_state.pdf_docs = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize Groq API on startup
if 'llm' not in st.session_state or st.session_state.llm is None:
    with st.spinner("Initializing Groq API..."):
        # Initialize LLM with predefined model
        st.session_state.llm = ChatGroq(
            temperature=0.3,
            model_name="llama3-8b-8192",
            api_key=os.environ["GROQ_API_KEY"]
        )
        
        # Initialize memory and conversation
        memory = ConversationBufferMemory()
        st.session_state.conversation = ConversationChain(
            llm=st.session_state.llm,
            memory=memory,
            verbose=True
        )

def generate_advanced_data_summary(df):
    """Generate an advanced summary of the sales data"""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    total_sales = df['Sales'].sum()
    avg_sale = df['Sales'].mean()
    median_sale = df['Sales'].median()
    sales_std = df['Sales'].std()

    monthly_sales = df.groupby('Month', observed=False)['Sales'].sum()
    best_month, worst_month = monthly_sales.idxmax(), monthly_sales.idxmin()

    product_stats = df.groupby('Product', observed=False)['Sales'].agg(['sum', 'count'])
    top_product, most_freq_product = product_stats['sum'].idxmax(), product_stats['count'].idxmax()

    region_sales = df.groupby('Region', observed=False)['Sales'].sum()
    best_region, worst_region = region_sales.idxmax(), region_sales.idxmin()

    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, labels=age_labels, right=False)
    best_age_group = df.groupby('Age_Group', observed=False)['Sales'].mean().idxmax()

    gender_sales = df.groupby('Customer_Gender', observed=False)['Sales'].mean()

    summary = f'''
Advanced Sales Data Summary:

• **Total sales** : ${total_sales:,.2f}
• **Average / Median sale** : ${avg_sale:.2f} / ${median_sale:.2f}
• **σ(Sales)** : ${sales_std:.2f}

_Time window_
  – Best month : {best_month}  Worst month : {worst_month}

_Product_
  – Highest revenue : {top_product}  Most frequently sold : {most_freq_product}

_Regions_
  – Best : {best_region}  Worst : {worst_region}

_Customers_
  – Avg satisfaction : {df.Customer_Satisfaction.mean():.2f} (±{df.Customer_Satisfaction.std():.2f})
  – Best age group : {best_age_group}
  – Avg sale by gender : Male ${gender_sales.get("Male", 0):.2f} / Female ${gender_sales.get("Female", 0):.2f}
'''
    return summary

def create_vector_db():
    """Create a vector database from the texts"""
    with st.spinner("Creating vector database..."):
        # Using SentenceTransformers embeddings 
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.texts, embeddings)
        st.success("Vector database created successfully!")

def setup_rag_system():
    """Set up the RAG system"""
    if st.session_state.vectorstore is None:
        st.warning("Vector database not created yet. Please create it first.")
        return
    
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create RetrievalQA chain
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        retriever=st.session_state.retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    # Set up Wikipedia API wrapper
    st.session_state.wikipedia_wrapper = WikipediaAPIWrapper()
    
    # Create Wikipedia tool
    st.session_state.wikipedia_tool = Tool(
        name="Wikipedia Search",
        func=wiki_search,
        description="Searches Wikipedia for information"
    )
    
    # Create data analysis and recommendation chains
    data_analysis_template = """
    Analyze the following advanced sales data summary:

    {advanced_summary}

    Provide a concise analysis of the key points:
    """
    data_analysis_prompt = PromptTemplate(template=data_analysis_template, input_variables=["advanced_summary"])
    data_analysis_chain = LLMChain(llm=st.session_state.llm, prompt=data_analysis_prompt, output_key="analysis")

    recommendation_template = """
    Based on the following analysis of sales data:

    {analysis}

    Provide specific recommendations to address the question: {question}

    Recommendations:
    """
    recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=["analysis", "question"])
    recommendation_chain = LLMChain(llm=st.session_state.llm, prompt=recommendation_prompt, output_key="recommendations")

    st.session_state.overall_chain = SequentialChain(
        chains=[data_analysis_chain, recommendation_chain],
        input_variables=["advanced_summary", "question"],
        output_variables=["analysis", "recommendations"]
    )
    
    st.success("RAG system setup complete!")

def wiki_search(query):
    """Search Wikipedia for information"""
    try:
        # Use the wrapper to get content
        content = st.session_state.wikipedia_wrapper.run(query)

        # Use Wikipedia search to get URLs
        search_results = wikipedia.search(query, results=3)
        urls = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                # Specify the parser explicitly
                soup = BeautifulSoup(page.html(), features="lxml")
                content += f"\nTitle: {page.title}\nSummary: {soup.get_text()[:500]}...\n"
                urls.append(page.url)
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
        return {'content': content, 'urls': urls}
    except Exception as e:
        return {'content': f"An error occurred: {str(e)}", 'urls': []}

def generate_rag_insight(question):
    """Generate insights using RAG system"""
    if "qa_chain" not in st.session_state:
        st.warning("RAG system not set up yet.")
        return "Please set up the RAG system first."
        
    # First, use the existing retriever to get relevant documents
    context = f"Advanced Sales Summary:\n{st.session_state.advanced_summary}\n\nQuestion: {question}"
    with st.spinner("Generating RAG insights..."):
        result = st.session_state.qa_chain({"query": context})
        
        # Get Wikipedia content and URLs related to the question
        wiki_results = st.session_state.wikipedia_tool.run(question)
        wiki_content = wiki_results['content']
        wiki_urls = wiki_results['urls']
        
        # Combine the existing context with Wikipedia results
        enhanced_context = f"{context}\n\nAdditional information from Wikipedia:\n{wiki_content}"
        
        # Use the enhanced context to generate the final insight
        final_result = st.session_state.qa_chain({"query": enhanced_context})
        
        insight = final_result['result']
        sources = [doc.metadata['source'] for doc in final_result['source_documents']]
        
        # Add Wikipedia to the sources, including specific URLs
        sources.extend([f"Wikipedia: {url}" for url in wiki_urls])
        
        return f"Insight:\n{insight}\n\nSources:\n" + "\n".join(set(sources))

def generate_chained_insight(question):
    """Generate insights using sequential chain"""
    if "overall_chain" not in st.session_state:
        st.warning("Sequential chain not set up yet.")
        return "Please set up the RAG system first."
    
    with st.spinner("Generating chained insights..."):
        try:
            result = st.session_state.overall_chain({"advanced_summary": st.session_state.advanced_summary, "question": question})
            return f"Analysis:\n{result['analysis']}\n\nRecommendations:\n{result['recommendations']}"
        except Exception as e:
            return f"Error: {str(e)}"

def setup_agent():
    """Set up the agent"""
    if st.session_state.llm is None:
        st.warning("LLM not initialized yet.")
        return
    
    # Create tools for agent
    def plot_product_category_sales():
        product_cat_sales = st.session_state.df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        product_cat_sales.plot(kind='bar', ax=ax)
        ax.set_title('Sales Distribution by Product')
        ax.set_xlabel('Product')
        ax.set_ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_sales_trend():
        fig, ax = plt.subplots(figsize=(10, 6))
        st.session_state.df.groupby('Date')['Sales'].sum().plot(ax=ax)
        ax.set_title('Daily Sales Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Sales')
        plt.tight_layout()
        return fig
    
    # Function to get sales summary
    def get_sales_summary(query=None):
        return st.session_state.advanced_summary
    
    # Create tools
    sales_plot = Tool(
        name="ProductCategorySalesPlot",
        func=lambda x: plot_product_category_sales(),
        description="Generates a plot of sales distribution by product category"
    )
    
    sales_trend = Tool(
        name="SalesTrendPlot",
        func=lambda x: plot_sales_trend(),
        description="Generates a plot of the daily sales trend"
    )
    
    knowledge_tool = Tool(
        name="RAGInsight",
        func=generate_rag_insight,
        description="Generates insights using RAG system"
    )
    
    sales_summary_tool = Tool(
        name="Sales Data Summary",
        func=get_sales_summary,
        description="Provides a comprehensive summary of sales data, including top products, regions, and customer demographics."
    )
    
    tools = [sales_plot, sales_trend, knowledge_tool, sales_summary_tool]
    
    # Create agent
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=st.session_state.llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    st.success("Agent setup complete!")

def process_pdfs(uploaded_files):
    """Process uploaded PDF files"""
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []

    for uploaded_file in uploaded_files:
        # Save uploaded file to temporary directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)
    
    st.session_state.pdf_docs = []
    
    # Load PDFs
    with st.spinner("Processing PDF files..."):
        for file_path in pdf_paths:
            loader = PyPDFLoader(file_path)
            st.session_state.pdf_docs.extend(loader.load())
    
    st.success(f"Loaded {len(st.session_state.pdf_docs)} pages from PDFs")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.texts = text_splitter.split_documents(st.session_state.pdf_docs)
    
    # Add the sales summary document if it exists
    if st.session_state.sales_summary_doc:
        st.session_state.texts.append(st.session_state.sales_summary_doc)
    
    st.success(f"Created {len(st.session_state.texts)} text chunks")

# Home page
if page == "Home":
    st.title("Sales Data Analysis Dashboard")
    st.markdown("""
    Welcome to the Sales Data Analysis Dashboard. This application uses Groq API with LLama3 to analyze sales data and generate insights.
    
    ### Features:
    - Upload and analyze sales data
    - Create a knowledge base from PDF documents
    - Generate insights using RAG system
    - Interact with an agent for sales analysis
    
    ### Getting Started:
    1. Navigate to the Data Upload page
    2. Upload your sales data and PDF documents
    3. Start analyzing!
    """)

# Data Upload page
elif page == "Data Upload":
    st.title("Data Upload")
    
    # CSV Upload
    st.header("Upload Sales Data (CSV)")
    uploaded_csv = st.file_uploader("Upload sales data CSV file", type=["csv"])
    
    if uploaded_csv:
        try:
            st.session_state.df = pd.read_csv(uploaded_csv)
            st.success("CSV file uploaded successfully!")
            st.dataframe(st.session_state.df.head())
            
            # Generate advanced summary
            if st.button("Generate Advanced Summary"):
                st.session_state.advanced_summary = generate_advanced_data_summary(st.session_state.df)
                
                # Create sales summary document
                st.session_state.sales_summary_doc = Document(
                    page_content=st.session_state.advanced_summary,
                    metadata={"source": "sales_summary", "title": "Advanced Sales Data Summary"}
                )
                
                st.success("Advanced summary generated!")
                st.markdown(st.session_state.advanced_summary)
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # PDF Upload
    st.header("Upload PDF Documents")
    uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_pdfs:
        if st.button("Process PDF Files"):
            process_pdfs(uploaded_pdfs)
    
    # Create Vector DB
    if st.session_state.texts and len(st.session_state.texts) > 0:
        if st.button("Create Vector Database"):
            create_vector_db()

# Data Analysis page
elif page == "Data Analysis":
    st.title("Data Analysis")
    
    if st.session_state.df is None:
        st.warning("Please upload sales data first on the Data Upload page.")
    else:
        st.header("Sales Data Overview")
        st.dataframe(st.session_state.df.head())
        
        if st.session_state.advanced_summary:
            st.header("Advanced Sales Summary")
            st.markdown(st.session_state.advanced_summary)
        
        st.header("Data Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Product Sales", "Sales Trend", "Regional Analysis", "Customer Analysis"])
        
        with tab1:
            st.subheader("Product Sales Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            product_sales = st.session_state.df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
            product_sales.plot(kind='bar', ax=ax)
            ax.set_title('Sales by Product')
            ax.set_xlabel('Product')
            ax.set_ylabel('Total Sales')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Sales Trend Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            st.session_state.df.groupby('Date')['Sales'].sum().plot(ax=ax)
            ax.set_title('Daily Sales Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Sales')
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Regional Sales Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            region_sales = st.session_state.df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            region_sales.plot(kind='bar', ax=ax)
            ax.set_title('Sales by Region')
            ax.set_xlabel('Region')
            ax.set_ylabel('Total Sales')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab4:
            st.subheader("Customer Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Sales by Age Group")
                age_bins = [0, 25, 35, 45, 55, 100]
                age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
                st.session_state.df['Age_Group'] = pd.cut(st.session_state.df['Customer_Age'], bins=age_bins, labels=age_labels, right=False)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                st.session_state.df.groupby('Age_Group')['Sales'].mean().plot(kind='bar', ax=ax)
                ax.set_title('Average Sales by Age Group')
                ax.set_xlabel('Age Group')
                ax.set_ylabel('Average Sales')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.write("Sales by Gender")
                fig, ax = plt.subplots(figsize=(8, 6))
                st.session_state.df.groupby('Customer_Gender')['Sales'].mean().plot(kind='bar', ax=ax)
                ax.set_title('Average Sales by Gender')
                ax.set_xlabel('Gender')
                ax.set_ylabel('Average Sales')
                plt.tight_layout()
                st.pyplot(fig)

# RAG System page
elif page == "RAG System":
    st.title("RAG System")
    
    if st.session_state.vectorstore is None:
        st.warning("Vector database not created yet. Please create it on the Data Upload page.")
    else:
        # Automatically set up the RAG system if not already done
        if "retriever" not in st.session_state:
            with st.spinner("Setting up the RAG System..."):
                setup_rag_system()
            st.success("RAG System is ready!")
        
        st.header("Generate Insights")
        
        # Create tabs for different insight methods
        tab1, tab2 = st.tabs(["RAG Insights", "Sequential Chain Insights"])
        
        with tab1:
            st.subheader("RAG Insights")
            question = st.text_input("Enter your question for RAG system", key="rag_question")
            
            if st.button("Generate RAG Insight"):
                if question:
                    insight = generate_rag_insight(question)
                    st.markdown(insight)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": question, "answer": insight})
                else:
                    st.warning("Please enter a question.")
        
        with tab2:
            st.subheader("Sequential Chain Insights")
            question = st.text_input("Enter your question for sequential chain", key="seq_question")
            
            if st.button("Generate Chained Insight"):
                if question:
                    insight = generate_chained_insight(question)
                    st.markdown(insight)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": question, "answer": insight})
                else:
                    st.warning("Please enter a question.")
        
        # Chat history
        if st.session_state.chat_history:
            st.header("Chat History")
            for i, chat in enumerate(st.session_state.chat_history):
                st.subheader(f"Question {i+1}: {chat['question']}")
                st.markdown(chat['answer'])
                st.divider()

# Agent Analysis page
elif page == "Agent Analysis":
    st.title("Agent Analysis")
    
    if st.session_state.vectorstore is None:
        st.warning("Vector database not created yet. Please create it on the Data Upload page.")
    else:
        # Automatically set up the agent if not already done
        if "agent" not in st.session_state or st.session_state.agent is None:
            with st.spinner("Setting up the Sales Analysis Agent..."):
                setup_agent()
            st.success("Agent is ready for your questions!")
        
        st.header("Interact with Sales Analysis Agent")
        
        question = st.text_input("Ask the agent a question about your sales data")
        
        if st.button("Ask Agent"):
            if question:
                with st.spinner("Agent is thinking..."):
                    try:
                        response = st.session_state.agent.run(question)
                        st.markdown(response)
                        
                        # Check if response contains plots
                        if "ProductCategorySalesPlot" in response:
                            fig = plot_product_category_sales()
                            st.pyplot(fig)
                        
                        if "SalesTrendPlot" in response:
                            fig = plot_sales_trend()
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Agent error: {str(e)}")
            else:
                st.warning("Please enter a question.")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit and powered by Llama3 via Groq API")

# Main function to run the app
if __name__ == "__main__":
    # The app is already running at this point
    pass