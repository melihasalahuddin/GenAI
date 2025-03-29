import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from typing import List, Dict, Any, Tuple

st.set_page_config(page_title="Data Analysis App", layout="wide")
st.title("Data Analysis App")

# Define a class for natural language query processing
class NaturalLanguageQueryProcessor:
    def __init__(self, dataframe=None):
        self.df = dataframe
        self.column_types = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        if dataframe is not None:
            self._analyze_columns()

    def set_dataframe(self, dataframe):
        self.df = dataframe
        self._analyze_columns()

    def _analyze_columns(self):
        if self.df is None:
            return
        
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Try to identify date columns
        self.date_columns = []
        for col in self.df.columns:
            if col in self.numeric_columns or col in self.categorical_columns:
                continue
            try:
                pd.to_datetime(self.df[col])
                self.date_columns.append(col)
            except:
                pass
        
        # Record column types
        self.column_types = {}
        for col in self.numeric_columns:
            self.column_types[col] = "numeric"
        for col in self.categorical_columns:
            self.column_types[col] = "categorical"
        for col in self.date_columns:
            self.column_types[col] = "date"

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query to determine the analysis intent"""
        query = query.lower()
        
        # Check if we have a dataframe
        if self.df is None:
            return {"error": "No data loaded. Please upload a file first."}
        
        # Check if query is related to the dataset
        if not self._is_data_related_query(query):
            return {
                "type": "non_data_query",
                "text": "I can only answer questions about the uploaded dataset. Please ask a question related to your data analysis."
            }
        
        # Check specific analysis types
        if self._is_summary_query(query):
            return self._handle_summary_query(query)
        elif self._is_count_query(query):
            return self._handle_count_query(query)
        elif self._is_average_query(query):
            return self._handle_average_query(query)
        elif self._is_distribution_query(query):
            return self._handle_distribution_query(query)
        elif self._is_correlation_query(query):
            return self._handle_correlation_query(query)
        elif self._is_top_bottom_query(query):
            return self._handle_top_bottom_query(query)
        elif self._is_missing_query(query):
            return self._handle_missing_query(query)
        elif self._is_filter_query(query):
            return self._handle_filter_query(query)
        else:
            # Generic response if we can't determine specific intent but it seems data-related
            return {
                "type": "general_info",
                "text": f"Here's a summary of your data. It has {self.df.shape[0]} rows and {self.df.shape[1]} columns.",
                "data": self.df.head(5)
            }

    # Query type detection methods
    def _is_data_related_query(self, query: str) -> bool:
        """Check if the query is related to data analysis of the uploaded dataset"""
        # List of keywords that indicate a data-related question
        data_keywords = [
            'data', 'dataset', 'dataframe', 'file', 'csv', 'excel', 'column', 'row', 
            'value', 'average', 'mean', 'median', 'mode', 'sum', 'count', 'total',
            'maximum', 'minimum', 'max', 'min', 'correlation', 'relationship',
            'distribution', 'histogram', 'bar chart', 'plot', 'graph', 'visualization',
            'missing', 'null', 'na', 'nan', 'empty', 'statistic', 'stat', 'stats',
            'describe', 'summary', 'overview', 'analysis', 'analyze', 'show', 'display',
            'find', 'filter', 'sort', 'top', 'bottom', 'highest', 'lowest'
        ]
        
        # Check if any column name is mentioned in the query
        column_mentioned = any(col.lower() in query for col in self.df.columns if isinstance(col, str))
        
        # Check if any data keyword is present in the query
        keyword_present = any(keyword in query for keyword in data_keywords)
        
        # If either condition is met, consider it a data-related query
        return column_mentioned or keyword_present
    
    def _is_summary_query(self, query: str) -> bool:
        keywords = ['summary', 'describe', 'overview', 'statistics', 'stats', 'tell me about']
        return any(keyword in query for keyword in keywords)
    
    def _is_count_query(self, query: str) -> bool:
        return re.search(r'how many|count|total number', query) is not None
    
    def _is_average_query(self, query: str) -> bool:
        return re.search(r'average|mean|median|typical', query) is not None
    
    def _is_distribution_query(self, query: str) -> bool:
        return re.search(r'distribution|histogram|spread|range', query) is not None
    
    def _is_correlation_query(self, query: str) -> bool:
        return re.search(r'correlation|relationship|compare|versus|vs', query) is not None
    
    def _is_top_bottom_query(self, query: str) -> bool:
        return re.search(r'top|bottom|highest|lowest|maximum|minimum|max|min', query) is not None
    
    def _is_missing_query(self, query: str) -> bool:
        return re.search(r'missing|null|empty|na', query) is not None
    
    def _is_filter_query(self, query: str) -> bool:
        return re.search(r'where|filter|show me|find|search', query) is not None

    # Query handlers
    def _handle_summary_query(self, query: str) -> Dict[str, Any]:
        # Identify if query is about specific columns
        columns = self._extract_columns_from_query(query)
        
        if not columns:
            # General summary of all numeric columns
            result = {
                "type": "summary",
                "text": "Here's a summary of the numeric columns in your data:",
                "data": self.df.describe()
            }
        else:
            # Summary of specific columns
            try:
                result = {
                    "type": "summary",
                    "text": f"Here's a summary of the column(s) {', '.join(columns)}:",
                    "data": self.df[columns].describe() if all(col in self.numeric_columns for col in columns) else self.df[columns].head(10)
                }
            except Exception as e:
                result = {"error": f"Error generating summary: {str(e)}"}
        
        return result

    def _handle_count_query(self, query: str) -> Dict[str, Any]:
        # Try to find what to count
        columns = self._extract_columns_from_query(query)
        
        if "rows" in query or "entries" in query or "records" in query:
            return {
                "type": "count",
                "text": f"There are {self.df.shape[0]} rows in your data.",
                "data": self.df.shape[0]
            }
        elif "columns" in query or "fields" in query:
            return {
                "type": "count",
                "text": f"There are {self.df.shape[1]} columns in your data.",
                "data": self.df.columns.tolist()
            }
        elif columns:
            # Count values in a categorical column
            if len(columns) == 1 and columns[0] in self.categorical_columns:
                value_counts = self.df[columns[0]].value_counts()
                return {
                    "type": "count_values",
                    "text": f"Here's the count of values in column '{columns[0]}':",
                    "data": value_counts,
                    "chart": "bar"
                }
            else:
                return {
                    "type": "count",
                    "text": f"Not sure what to count related to {', '.join(columns)}. Here's some basic information:",
                    "data": self.df[columns].head(10) if all(col in self.df.columns for col in columns) else self.df.head(5)
                }
        else:
            return {
                "type": "count", 
                "text": f"Your data has {self.df.shape[0]} rows and {self.df.shape[1]} columns.",
                "data": {"rows": self.df.shape[0], "columns": self.df.shape[1]}
            }

    def _handle_average_query(self, query: str) -> Dict[str, Any]:
        columns = self._extract_columns_from_query(query)
        
        # Filter to only include numeric columns
        valid_columns = [col for col in columns if col in self.numeric_columns]
        
        if not valid_columns:
            # If no specific column was mentioned or valid, show averages for all numeric columns
            result = {
                "type": "average",
                "text": "Here are the average values for all numeric columns:",
                "data": self.df[self.numeric_columns].mean()
            }
        else:
            # Average of specific columns
            try:
                result = {
                    "type": "average",
                    "text": f"Here are the average values for column(s) {', '.join(valid_columns)}:",
                    "data": self.df[valid_columns].mean()
                }
            except Exception as e:
                result = {"error": f"Error calculating average: {str(e)}"}
        
        return result

    def _handle_distribution_query(self, query: str) -> Dict[str, Any]:
        columns = self._extract_columns_from_query(query)
        
        if not columns:
            # If no column specified, suggest looking at distribution of a numeric column
            if self.numeric_columns:
                column = self.numeric_columns[0]
                result = {
                    "type": "distribution",
                    "text": f"Here's the distribution of values in column '{column}':",
                    "data": self.df[column].value_counts(),
                    "column": column,
                    "chart_type": "distribution"
                }
            else:
                result = {"error": "No numeric columns found to show distribution."}
        else:
            # Show distribution of the specified column
            if columns[0] in self.numeric_columns:
                # For numeric columns, create bins and count
                column = columns[0]
                try:
                    # Create a simple binned distribution
                    min_val = self.df[column].min()
                    max_val = self.df[column].max()
                    bins = 10
                    bin_edges = np.linspace(min_val, max_val, bins + 1)
                    labels = [f'{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}' for i in range(bins)]
                    binned = pd.cut(self.df[column], bins=bin_edges, labels=labels)
                    distribution = binned.value_counts().sort_index()
                    
                    result = {
                        "type": "distribution",
                        "text": f"Here's the distribution of values in numeric column '{column}':",
                        "data": distribution,
                        "column": column,
                        "chart_type": "distribution"
                    }
                except Exception as e:
                    result = {"error": f"Error creating distribution: {str(e)}"}
            elif columns[0] in self.categorical_columns:
                result = {
                    "type": "distribution",
                    "text": f"Here's the distribution of values in categorical column '{columns[0]}':",
                    "data": self.df[columns[0]].value_counts(),
                    "column": columns[0],
                    "chart_type": "distribution"
                }
            else:
                result = {"error": f"Column '{columns[0]}' not found or not suitable for distribution analysis."}
        
        return result

    def _handle_correlation_query(self, query: str) -> Dict[str, Any]:
        columns = self._extract_columns_from_query(query)
        
        if len(columns) >= 2:
            # Correlation between specific columns
            valid_columns = [col for col in columns if col in self.numeric_columns]
            if len(valid_columns) >= 2:
                corr_result = self.df[valid_columns].corr()
                return {
                    "type": "correlation",
                    "text": f"Here's the correlation between {', '.join(valid_columns)}:",
                    "data": corr_result,
                    "columns": valid_columns,
                    "chart_type": "correlation"
                }
            else:
                return {"error": "The specified columns are not numeric or not found."}
        else:
            # General correlation matrix
            if len(self.numeric_columns) >= 2:
                corr_matrix = self.df[self.numeric_columns].corr()
                return {
                    "type": "correlation",
                    "text": "Here's the correlation matrix for numeric columns:",
                    "data": corr_matrix,
                    "chart_type": "correlation"
                }
            else:
                return {"error": "Not enough numeric columns to calculate correlations."}

    def _handle_top_bottom_query(self, query: str) -> Dict[str, Any]:
        columns = self._extract_columns_from_query(query)
        
        # Determine if looking for top or bottom values
        is_top = re.search(r'top|highest|maximum|max', query) is not None
        n = self._extract_number_from_query(query) or 5  # Default to top/bottom 5
        
        if not columns:
            return {"error": "Please specify which column to find top/bottom values for."}
        
        try:
            column = columns[0]
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found."}
            
            if is_top:
                result = self.df.nlargest(n, column)
                return {
                    "type": "top",
                    "text": f"Top {n} values for column '{column}':",
                    "data": result
                }
            else:
                result = self.df.nsmallest(n, column)
                return {
                    "type": "bottom",
                    "text": f"Bottom {n} values for column '{column}':",
                    "data": result
                }
        except Exception as e:
            return {"error": f"Error processing top/bottom query: {str(e)}"}

    def _handle_missing_query(self, query: str) -> Dict[str, Any]:
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_counts,
            'Percentage (%)': missing_percent
        }).sort_values('Missing Values', ascending=False)
        
        return {
            "type": "missing",
            "text": "Here's the missing values analysis:",
            "data": missing_df,
            "chart_type": "missing"
        }

    def _handle_filter_query(self, query: str) -> Dict[str, Any]:
        # This is a simplified filter handler - a production version would need NLP
        columns = self._extract_columns_from_query(query)
        
        if not columns:
            return {
                "type": "general_info",
                "text": "I couldn't determine which columns to filter. Here's a preview of your data:",
                "data": self.df.head(5)
            }
        
        # Try to extract conditions - very simplified approach
        conditions = []
        try:
            # Simple patterns like "where column > value"
            for column in columns:
                if column not in self.df.columns:
                    continue
                    
                # Look for column and comparison patterns
                comparison_pattern = rf"{column}\s*(>|<|==|>=|<=|=|!=)\s*(\d+)"
                match = re.search(comparison_pattern, query)
                
                if match:
                    operator = match.group(1)
                    if operator == '=':  # Convert = to ==
                        operator = '=='
                    value = float(match.group(2))
                    conditions.append((column, operator, value))
                
                # Check for values contained in column
                contains_pattern = rf"{column}\s*(contains|has|includes|with)\s*['\"]*([a-zA-Z0-9 ]+)['\"]*"
                match = re.search(contains_pattern, query)
                
                if match:
                    term = match.group(2).strip()
                    if column in self.categorical_columns:
                        conditions.append((column, 'contains', term))
        
            # Apply filters
            if conditions:
                filtered_df = self.df.copy()
                for col, op, val in conditions:
                    if op == '==':
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == '>':
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == '<':
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == '>=':
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == '<=':
                        filtered_df = filtered_df[filtered_df[col] <= val]
                    elif op == '!=':
                        filtered_df = filtered_df[filtered_df[col] != val]
                    elif op == 'contains':
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val, case=False)]
                
                return {
                    "type": "filter",
                    "text": f"Filtered data based on condition(s): {', '.join([f'{c[0]} {c[1]} {c[2]}' for c in conditions])}",
                    "data": filtered_df.head(10)
                }
            else:
                # If no conditions extracted but columns specified, show those columns
                return {
                    "type": "column_view",
                    "text": f"Here's data for the column(s) {', '.join(columns)}:",
                    "data": self.df[columns].head(10)
                }
        except Exception as e:
            return {"error": f"Error filtering data: {str(e)}"}

    # Helper methods
    def _extract_columns_from_query(self, query: str) -> List[str]:
        """Extract column names from the query based on the dataframe columns"""
        if self.df is None:
            return []
        
        # Look for exact column name matches
        columns = []
        for col in self.df.columns:
            # Look for the column name as a whole word
            if re.search(r'\b' + re.escape(str(col).lower()) + r'\b', query.lower()):
                columns.append(col)
        
        return columns
    
    def _extract_number_from_query(self, query: str) -> int:
        """Extract a number from the query"""
        match = re.search(r'\b(\d+)\b', query)
        if match:
            return int(match.group(1))
        return None


# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None

# Initialize query processor
if "query_processor" not in st.session_state:
    st.session_state.query_processor = NaturalLanguageQueryProcessor()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# File upload and data info in the first column
with col1:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.session_state.query_processor.set_dataframe(data)
            
            st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            # Add data info
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            with st.expander("Data Information"):
                st.write(f"**Rows:** {data.shape[0]}")
                st.write(f"**Columns:** {data.shape[1]}")
                st.write(f"**Numeric Columns:** {len(numeric_cols)}")
                st.write(f"**Categorical Columns:** {len(categorical_cols)}")
                
                # Display column types
                col_df = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes,
                    'Non-Null Count': data.count(),
                    'Null Count': data.isnull().sum()
                })
                st.dataframe(col_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Natural language interface in the second column
with col2:
    st.header("Ask Questions About Your Data")
    
    if st.session_state.data is None:
        st.info("Please upload a data file to start asking questions.")
    else:
        # Display example queries
        with st.expander("Example Questions You Can Ask"):
            st.markdown("""
            - "Show me a summary of the data"
            - "What's the average of [column]?"
            - "Show the distribution of [column]"
            - "How many rows are in the dataset?"
            - "What's the correlation between [column1] and [column2]?"
            - "Show me the top 5 values in [column]"
            - "How many missing values are there?"
            - "Show me data where [column] > 50"
            """)
            
        st.info("⚠️ Note: This app can only answer questions about the uploaded dataset. It cannot respond to general knowledge questions or topics unrelated to data analysis.")
        
        # Query input
        query = st.text_input("Ask a question about your data")
        
        if query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Process the query
            result = st.session_state.query_processor.parse_query(query)
            
            # Format and add the response to chat history
            if "error" in result:
                response = {"role": "assistant", "content": result["error"], "data": None}
            else:
                response = {
                    "role": "assistant", 
                    "content": result["text"], 
                    "data": result.get("data"),
                    "chart_type": result.get("chart_type"),
                    "column": result.get("column"),
                    "columns": result.get("columns"),
                    "type": result.get("type")
                }
            
            st.session_state.chat_history.append(response)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Assistant:** {message['content']}")
                
                # Only display data or charts if it's not a non-data query
                if message.get("type") != "non_data_query":
                    # Display data if available
                    if message.get("data") is not None:
                        if isinstance(message["data"], pd.DataFrame):
                            st.dataframe(message["data"], use_container_width=True)
                        elif isinstance(message["data"], pd.Series):
                            st.dataframe(message["data"].to_frame(), use_container_width=True)
                        else:
                            st.write(message["data"])
                    
                    # Create streamlit native charts instead of matplotlib
                    chart_type = message.get("chart_type")
                    if chart_type == "distribution":
                        column = message.get("column")
                        if column and isinstance(message.get("data"), pd.Series):
                            st.subheader(f"Distribution of {column}")
                            st.bar_chart(message["data"])
                    
                    elif chart_type == "correlation":
                        if isinstance(message.get("data"), pd.DataFrame):
                            st.subheader("Correlation Matrix")
                            # Display as a heatmap using a dataframe with styling
                            corr_data = message["data"]
                            st.dataframe(corr_data.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)
                    
                    elif chart_type == "missing":
                        if isinstance(message.get("data"), pd.DataFrame) and "Missing Values" in message["data"].columns:
                            st.subheader("Missing Values by Column")
                            chart_data = message["data"]["Missing Values"]
                            st.bar_chart(chart_data)
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()

# Add download functionality
if st.session_state.data is not None:
    with st.sidebar:
        st.header("Download Data")
        csv = st.session_state.data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="data_export.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
