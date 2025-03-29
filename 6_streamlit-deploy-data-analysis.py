import streamlit as st
import pandas as pd
import numpy as np

st.title("Data Analysis App")

# Initialize data storage
if "data" not in st.session_state:
    st.session_state.data = None

# Initialize analysis history
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

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
            
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Main content area
st.header("Data Analysis")

if st.session_state.data is not None:
    data = st.session_state.data
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Basic Statistics", "Missing Values", "Correlation Analysis", "Column Distribution", "Custom Query"]
    )
    
    if analysis_type == "Basic Statistics":
        st.subheader("Basic Statistics")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe())
        else:
            st.info("No numeric columns found for statistical analysis.")
    
    elif analysis_type == "Missing Values":
        st.subheader("Missing Values Analysis")
        missing_data = data.isnull().sum()
        missing_percent = (data.isnull().sum() / len(data) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage (%)': missing_percent
        })
        
        st.dataframe(missing_df)
        
        # Visualize missing values
        st.subheader("Missing Values Visualization")
        st.bar_chart(missing_df['Missing Values'])
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        numeric_data = data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            corr = numeric_data.corr()
            st.dataframe(corr)
            
            # Correlation heatmap
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for correlation analysis.")
    
    elif analysis_type == "Column Distribution":
        st.subheader("Column Distribution")
        
        # Select column to analyze
        column = st.selectbox("Select Column", data.columns)
        
        if column:
            st.write(f"### Distribution of {column}")
            
            # Check if column is numeric or categorical
            if pd.api.types.is_numeric_dtype(data[column]):
                # Numeric column
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                # Histogram
                data[column].hist(ax=ax[0], bins=30)
                ax[0].set_title("Histogram")
                
                # Box plot
                data.boxplot(column=[column], ax=ax[1])
                ax[1].set_title("Box Plot")
                
                st.pyplot(fig)
                
                # Summary statistics
                st.write("### Summary Statistics")
                stats = data[column].describe()
                st.dataframe(pd.DataFrame(stats).T)
                
            else:
                # Categorical column
                value_counts = data[column].value_counts()
                
                # Bar chart
                st.bar_chart(value_counts)
                
                # Value counts table
                st.write("### Value Counts")
                st.dataframe(pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage (%)': (value_counts.values / len(data) * 100).round(2)
                }))
    
    elif analysis_type == "Custom Query":
        st.subheader("Custom SQL-like Query")
        st.write("Enter a Python query using the variable 'data' to represent your dataframe.")
        st.write("Example: data[data['column'] > 50].head()")
        
        query = st.text_area("Enter your query")
        
        if st.button("Run Query"):
            try:
                # Execute the custom query
                result = eval(query)
                
                # Display the result
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                else:
                    st.write(result)
                
                # Add to history
                st.session_state.analysis_history.append({
                    "query": query,
                    "type": "custom"
                })
            except Exception as e:
                st.error(f"Error executing query: {e}")
    
    # Download analyzed data
    st.header("Download Data")
    
    if st.button("Generate Download Link"):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analyzed_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info("Please upload a data file using the sidebar to begin analysis.")

# Load required libraries only if needed for visualization
if st.session_state.data is not None and "Correlation Analysis" in st.session_state and st.session_state["Correlation Analysis"]:
    import matplotlib.pyplot as plt
    import seaborn as sns

# For CSV download functionality
import base64
