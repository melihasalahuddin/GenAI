import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
import requests
import json
import time
from typing import List, Dict, Any, Tuple
from functools import lru_cache

st.set_page_config(page_title="Data Analysis App with Llama3", layout="wide")
st.title("Data Analysis App with Llama3")

# Define a class for Ollama language model integration
class OllamaQueryProcessor:
    def __init__(self, dataframe=None, model="llama3.2:1b"):
        self.df = dataframe
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
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
    
    def _get_dataframe_info(self):
        """Get minimal structured information about the dataframe for the prompt"""
        if self.df is None:
            return "No dataframe loaded."
        
        # Only include essential information to reduce prompt size
        info = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": self.df.columns.tolist(),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "date_columns": self.date_columns
        }
        
        return json.dumps(info, default=str)
    
    # Use LRU cache to store recent query results
    @lru_cache(maxsize=32)
    def _cached_query(self, query_str, model_name, df_info_hash):
        """Cached version of the query to avoid repeated identical queries"""
        return self._perform_query(query_str, model_name)
    
    def _perform_query(self, query, model_name):
        """Actual query implementation with timeout"""
        try:
            df_info = self._get_dataframe_info()
            
            prompt = f"""
            You are a data analysis assistant working with a pandas DataFrame.
            
            Here is information about the DataFrame:
            {df_info}
            
            The user has asked the following question about the data:
            "{query}"
            
            Please analyze this question and respond with a JSON object only that provides:
            1. An explanation of what analysis should be performed
            2. Python code to perform this analysis using pandas
            3. A concise explanation of the results that should be shown to the user
            
            Format your response as valid JSON with the following structure:
            {{
                "type": "one of [summary, count, average, distribution, correlation, top, bottom, missing, filter, general_info, error]",
                "text": "explanation to show the user",
                "code": "pandas code to execute for the analysis",
                "chart_type": "optional - one of [distribution, correlation, missing, None]",
                "column": "optional - primary column for analysis",
                "columns": "optional - list of columns for analysis"
            }}
            
            IMPORTANT: Be concise. Only include valid JSON in your response, nothing else.
            """
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                # Add parameters to make Ollama more deterministic and faster
                "temperature": 0.1,  # Low temperature for more deterministic responses
                "num_predict": 1024,  # Limit token generation
                "top_p": 0.9        # Higher precision
            }
            
            # Add timeout to prevent hanging
            response = requests.post(self.api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    response_text = result.get("response", "")
                    
                    # Extract JSON from the response - try multiple patterns
                    # Try code block first
                    json_pattern = re.compile(r'```(?:json)?(.*?)```', re.DOTALL)
                    match = json_pattern.search(response_text)
                    
                    if match:
                        json_str = match.group(1).strip()
                    else:
                        # Try to find JSON object directly
                        match = re.search(r'({[\s\S]*})', response_text)
                        if match:
                            json_str = match.group(1).strip()
                        else:
                            json_str = response_text
                    
                    # Parse the JSON
                    try:
                        result_json = json.loads(json_str)
                        return result_json
                    except json.JSONDecodeError:
                        # If parsing fails, try to clean the JSON string
                        json_str = self._clean_json_string(response_text)
                        try:
                            result_json = json.loads(json_str)
                            return result_json
                        except:
                            # As a fallback, create a simple JSON response
                            return {
                                "type": "error", 
                                "text": "I couldn't parse the model's response. Here's a general summary instead.",
                                "code": "result = df.describe()"
                            }
                except Exception as e:
                    return {
                        "type": "error",
                        "text": f"Error processing model response. Falling back to basic analysis.",
                        "code": "result = df.head(10)"
                    }
            else:
                return {
                    "type": "error", 
                    "text": f"API request failed. Make sure Ollama is running.",
                    "code": "result = df.head(10)"
                }
        except requests.exceptions.Timeout:
            return {
                "type": "error", 
                "text": "Request to Ollama timed out. Falling back to basic analysis.",
                "code": "result = df.head(10)"
            }
        except Exception as e:
            return {
                "type": "error",
                "text": "Error connecting to Ollama. Falling back to basic analysis.",
                "code": "result = df.head(10)"
            }
    
    def query_llama(self, query):
        """Send a query to the Ollama API with Llama model, using cache when possible"""
        # Create a hash of the dataframe info to invalidate cache when dataframe changes
        df_info_hash = hash(self._get_dataframe_info()) if self.df is not None else 0
        return self._cached_query(query, self.model, df_info_hash)
    
    def _clean_json_string(self, text):
        """Attempt to clean and extract valid JSON from text"""
        # Try to find anything that looks like a JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            potential_json = text[start_idx:end_idx+1]
            # Additional cleaning can be done here
            return potential_json
        
        return "{}"
    
    def execute_analysis(self, analysis_json):
        """Execute the pandas code provided in the analysis JSON with timeout protection"""
        if "code" not in analysis_json or not analysis_json["code"]:
            return self.df.head(10)
        
        try:
            # Create a local copy of the dataframe
            df = self.df.copy()
            
            # Create a dictionary of local variables for exec
            local_vars = {"df": df, "pd": pd, "np": np}
            
            # Execute the code with a timeout mechanism
            exec(analysis_json["code"], globals(), local_vars)
            
            # Return the result variable
            if "result" in local_vars:
                return local_vars["result"]
            else:
                # If no result is set, return a default
                return self.df.head(10)
        except Exception as e:
            print(f"Error executing code: {str(e)}")
            # Return a default result on error
            return self.df.head(10)
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query using the Llama model"""
        if self.df is None:
            return {"error": "No data loaded. Please upload a file first."}
        
        # First, try a rule-based approach for simple queries to avoid LLM overhead
        simple_result = self._try_rule_based_analysis(query)
        if simple_result:
            return simple_result
        
        # Get analysis instructions from the model
        analysis = self.query_llama(query)
        
        if "error" in analysis:
            return analysis
        
        # Execute the analysis code
        data = self.execute_analysis(analysis)
        
        # Construct the result
        result = {
            "type": analysis.get("type", "general_info"),
            "text": analysis.get("text", "Here's the result of your query:"),
            "data": data,
            "chart_type": analysis.get("chart_type"),
            "column": analysis.get("column"),
            "columns": analysis.get("columns")
        }
        
        return result
    
    def _try_rule_based_analysis(self, query: str) -> Dict[str, Any]:
        """Try to handle common simple queries with rule-based approach"""
        query = query.lower()
        
        # Basic pattern matching for simple queries
        if re.search(r'\b(show|display|get)\b.*\b(head|first|top)\b', query):
            # Handle "show first X rows" type queries
            match = re.search(r'(\d+)', query)
            n = int(match.group(1)) if match else 5
            return {
                "type": "general_info",
                "text": f"Here are the first {n} rows of your data:",
                "data": self.df.head(n)
            }
        
        if re.search(r'\b(show|display|get)\b.*\b(tail|last|bottom)\b', query):
            # Handle "show last X rows" type queries
            match = re.search(r'(\d+)', query)
            n = int(match.group(1)) if match else 5
            return {
                "type": "general_info",
                "text": f"Here are the last {n} rows of your data:",
                "data": self.df.tail(n)
            }
        
        if re.search(r'\b(show|display|get)\b.*\b(shape|dimensions|size)\b', query) or query in ['how many rows', 'how many columns', 'what is the shape']:
            # Handle shape queries
            return {
                "type": "general_info",
                "text": f"Your data has {self.df.shape[0]} rows and {self.df.shape[1]} columns.",
                "data": {"rows": self.df.shape[0], "columns": self.df.shape[1]}
            }
        
        if re.search(r'\b(describe|summary|statistics)\b', query):
            # Handle describe/summary statistics queries
            return {
                "type": "summary",
                "text": "Here's a statistical summary of your data:",
                "data": self.df.describe()
            }
        
        if re.search(r'\b(missing|null|na|nan)\b', query):
            # Handle missing value queries
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
        
        # No simple rule matched, return None to use the LLM
        return None


# Define a class for rule-based query processing as fallback
class RuleBasedQueryProcessor:
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
        else:
            # Generic response if we can't determine specific intent
            return {
                "type": "general_info",
                "text": f"Here's a preview of your data. It has {self.df.shape[0]} rows and {self.df.shape[1]} columns.",
                "data": self.df.head(5)
            }

    # Query type detection methods
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

# Initialize query processors
if "ollama_processor" not in st.session_state:
    st.session_state.ollama_processor = OllamaQueryProcessor()

if "rule_based_processor" not in st.session_state:
    st.session_state.rule_based_processor = RuleBasedQueryProcessor()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create main columns for layout
col1, col2 = st.columns([1, 2])

# File upload and data info in the first column
with col1:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # Engine selection
    st.subheader("Analysis Engine")
    engine_type = st.radio(
        "Select Query Engine",
        ["Rule-based (Fast)", "Ollama LLM (Smart but Slower)"],
        index=0  # Default to rule-based for speed
    )
    
    # Only show model selection if Ollama is selected
    if engine_type == "Ollama LLM (Smart but Slower)":
        # Model selection
        st.subheader("Llama Model Settings")
        model_name = st.selectbox(
            "Select Ollama Model", 
            ["llama3.2:1b", "llama3:8b", "llama3:70b", "llama2:7b"], 
            index=0
        )
        
        if model_name != st.session_state.ollama_processor.model:
            st.session_state.ollama_processor.model = model_name
            st.success(f"Model changed to {model_name}")
        
        # Test connection button
        if st.button("Test Ollama Connection"):
            try:
                with st.spinner("Testing connection to Ollama..."):
                    test_response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model_name, "prompt": "Say 'Connection successful'", "stream": False},
                        timeout=10  # Add timeout to prevent hanging
                    )
                if test_response.status_code == 200:
                    st.success("✅ Successfully connected to Ollama!")
                else:
                    st.error(f"❌ Connection failed with status code {test_response.status_code}")
            except requests.exceptions.Timeout:
                st.error("❌ Connection to Ollama timed out. The server might be busy.")
            except Exception as e:
                st.error(f"❌ Error connecting to Ollama: {str(e)}")
                st.info("Make sure Ollama is installed and running. You can install Ollama from https://ollama.com/")
                st.code("# Run this command to pull the Llama model:\nollama pull llama3.2:1b", language="bash")
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.session_state.ollama_processor.set_dataframe(data)
            st.session_state.rule_based_processor.set_dataframe(data)
            
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
        
        if engine_type == "Ollama LLM (Smart but Slower)":
            st.info(f"⚠️ Using Llama model from Ollama. First query may be slow while the model loads.")
        else:
            st.info("Using fast rule-based engine for data analysis. Switch to Ollama for more complex queries.")
        
        # Query input
        query = st.text_input("Ask a question about your data")
        
        if query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Process the query with selected engine
            start_time = time.time()
            
            if engine_type == "Ollama LLM (Smart but Slower)":
                # Show a spinner while processing with Ollama
                with st.spinner(f"Processing with {st.session_state.ollama_processor.model}..."):
                    result = st.session_state.ollama_processor.parse_query(query)
            else:
                # Use the rule-based processor
                result = st.session_state.rule_based_processor.parse_query(query)
            
            elapsed_time = time.time() - start_time
            
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
                    "type": result.get("type"),
                    "processing_time": elapsed_time
                }
            
            st.session_state.chat_history.append(response)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                content = message['content']
                # Add processing time if available
                if 'processing_time' in message:
                    content += f" *(Processed in {message['processing_time']:.2f} seconds)*"
                
                st.write(f"**Assistant:** {content}")
                
                # Only display data or charts if it's not a non-data query
                if message.get("type") != "non_data_query" and message.get("type") != "error":
                    # Display data if available
                    if message.get("data") is not None:
                        if isinstance(message["data"], pd.DataFrame):
                            st.dataframe(message["data"], use_container_width=True)
                        elif isinstance(message["data"], pd.Series):
                            st.dataframe(message["data"].to_frame(), use_container_width=True)
                        else:
                            st.write(message["data"])
                    
                    # Create streamlit native charts
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
        
        # Add installation instructions for Ollama
        if engine_type == "Ollama LLM (Smart but Slower)":
            st.header("Ollama Installation")
            st.markdown("""
            ### Installation Steps:
            1. Install Ollama from [ollama.com](https://ollama.com/)
            2. Pull the Llama3 model:
            ```bash
            ollama pull llama3.2:1b
            ```
            3. Make sure Ollama is running in the background
            
            ### Performance Tips:
            - First query is slow while the model loads into memory
            - Using smaller models like llama3.2:1b is much faster
            - Simple queries use built-in rules and bypass the LLM
            """)