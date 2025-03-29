import streamlit as st
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("Data Analysis Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("Act like a data analysis assistant. Help users understand their data."))

# Initialize data storage
if "data" not in st.session_state:
    st.session_state.data = None

# File upload section
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Basic statistics
            st.subheader("Data Summary")
            
            # Check for numeric columns to show statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("Numeric Columns Statistics:")
                st.dataframe(data[numeric_cols].describe())
            
            # Show column information
            st.subheader("Column Info")
            column_info = pd.DataFrame({
                'Data Type': data.dtypes,
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum(),
                'Unique Values': [data[col].nunique() for col in data.columns]
            })
            st.dataframe(column_info)
            
            # Add data loaded notification to chat
            data_message = f"Data loaded successfully. File: {uploaded_file.name}, Shape: {data.shape[0]} rows × {data.shape[1]} columns."
            st.session_state.messages.append(AIMessage(data_message))
            
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, SystemMessage):
        # System messages are not displayed to the user
        pass

# Create the bar where we can type messages
prompt = st.chat_input("Ask about your data or request analysis...")

# Helper function to run data analysis
def analyze_data(query, data):
    # Example analysis functions
    analysis_results = ""
    
    # Detect if we need to show data or run analysis
    query_lower = query.lower()
    
    if "describe" in query_lower or "summary" in query_lower:
        analysis_results += f"## Data Summary\n```\n{data.describe().to_string()}\n```\n\n"
    
    if "missing" in query_lower or "null" in query_lower:
        missing_data = data.isnull().sum()
        analysis_results += f"## Missing Values\n```\n{missing_data.to_string()}\n```\n\n"
        
    if "correlation" in query_lower:
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr = numeric_data.corr()
            analysis_results += f"## Correlation Matrix\n```\n{corr.to_string()}\n```\n\n"
        else:
            analysis_results += "No numeric columns found for correlation analysis.\n\n"
    
    if "head" in query_lower or "show" in query_lower:
        analysis_results += f"## Data Preview\n```\n{data.head().to_string()}\n```\n\n"
    
    # Return LLM response if no specific analysis was triggered
    if not analysis_results:
        return None
        
    return analysis_results

# Did the user submit a prompt?
if prompt:
    # Add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add message to history
    st.session_state.messages.append(HumanMessage(prompt))
    
    # Check if data is loaded and if the query is about data analysis
    analysis_result = None
    if st.session_state.data is not None:
        analysis_result = analyze_data(prompt, st.session_state.data)
    
    # If analysis was performed, show results
    if analysis_result:
        with st.chat_message("assistant"):
            st.markdown(analysis_result)
        st.session_state.messages.append(AIMessage(analysis_result))
    else:
        # If no specific analysis or no data loaded, use the LLM
        llm = ChatOllama(
            model="llama3.2:1b",
            temperature=0.7  # Reduced from 2 for more consistent responses
        )
        
        # Add context about data if available
        if st.session_state.data is not None:
            data_context = (
                f"The user has uploaded a dataset with {st.session_state.data.shape[0]} rows and "
                f"{st.session_state.data.shape[1]} columns. "
                f"Columns: {', '.join(st.session_state.data.columns.tolist())}. "
                "Provide guidance on how to analyze this data."
            )
            context_message = SystemMessage(data_context)
            messages_with_context = st.session_state.messages.copy()
            messages_with_context.insert(1, context_message)
            result = llm.invoke(messages_with_context).content
        else:
            result = llm.invoke(st.session_state.messages).content
        
        with st.chat_message("assistant"):
            st.markdown(result)
        
        st.session_state.messages.append(AIMessage(result))
