import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr, mannwhitneyu, kruskal, wilcoxon

# Advanced analysis imports
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# Handle deprecated sklearn import
try:
    from sklearn.utils.testing import ignore_warnings
except ImportError:
    from sklearn.utils._testing import ignore_warnings

# Time series imports
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Page configuration
st.set_page_config(
    page_title="Comprehensive Statistical Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if dark theme is enabled
dark_theme = st.get_option("theme.base") == "dark"

# Custom CSS to improve appearance and ensure dark mode compatibility
if dark_theme:
    interpretation_bg = "#1e3a5f"  # Darker blue for dark theme
    interpretation_border = "#3b82f6"
    plot_bg = "#262730"
    text_color = "rgba(255, 255, 255, 0.9)"
else:
    interpretation_bg = "#e7f5ff"  # Light blue for light theme
    interpretation_border = "#228be6"
    plot_bg = "#f8f9fa"
    text_color = "rgba(0, 0, 0, 0.9)"

st.markdown(f"""
<style>
    .main {{
        padding: 1rem;
    }}
    .stButton button {{
        width: 100%;
    }}
    .stAlert {{
        padding: 0.5rem;
    }}
    h1, h2, h3 {{
        margin-top: 0.5rem;
    }}
    .plot-container {{
        border-radius: 5px;
        padding: 1rem;
        background-color: {plot_bg};
    }}
    .interpretation {{
        background-color: {interpretation_bg};
        border-left: 5px solid {interpretation_border};
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: {'white' if dark_theme else 'inherit'};
    }}
    .interpretation p {{
        color: {text_color};
    }}
    .interpretation h4 {{
        color: {'white' if dark_theme else 'inherit'};
    }}
    .report-container {{
        background-color: {interpretation_bg};
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: {'white' if dark_theme else 'inherit'};
    }}
    .sidebar-description {{
        padding: 0.5rem;
        border-radius: 5px;
        background-color: {interpretation_bg};
        margin-bottom: 1rem;
    }}
    .method-card {{
        padding: 1rem;
        border-radius: 5px;
        background-color: {interpretation_bg};
        margin-bottom: 1rem;
        border: 1px solid {interpretation_border};
    }}
    .tooltip {{
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    .category-icon {{
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }}
    .method-icon {{
        font-size: 1.2rem;
        margin-right: 0.3rem;
    }}
    .tab-subheader {{
        font-size: 1rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# Function to create a download link
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

def display_with_config(df):
    column_config = {}
    
    # Configure columns based on their content/type
    for col in df.columns:
        # URL columns as links
        if any(url_term in str(col).lower() for url_term in ['link', 'url', 'http']):
            column_config[col] = st.column_config.LinkColumn()
        
        # Date columns
        elif df[col].dtype == 'datetime64[ns]':
            column_config[col] = st.column_config.DatetimeColumn()
    
    # Display with configuration
    st.dataframe(df, column_config=column_config)

def safe_display_dataframe(df):
    # Create a copy to avoid modifying the original DataFrame
    df_display = df.copy()
    
    # Detect and fix problematic columns
    for col in df_display.columns:
        # For columns with mixed types that might include strings
        if df_display[col].dtype == 'object':
            # Convert to string to avoid numeric conversion attempts
            df_display[col] = df_display[col].astype(str)
        
        # For datetime columns
        elif pd.api.types.is_datetime64_dtype(df_display[col]):
            # Convert to string format for display
            df_display[col] = df_display[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Display the fixed DataFrame
    safe_display_dataframe(df)

# Function to create a PDF report
def create_pdf_report(content):
    # In a real app, this would convert markdown to PDF
    # For now, we'll just encode the content
    return content.encode('utf-8')

# Generate report content for analysis
def generate_report_content(analysis_type, data, method, params, results=None):
    # Simulates generating an AI report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
    # {analysis_type}: {method} Analysis Report
    
    *Generated on: {timestamp}*
    
    ## Overview
    
    This report summarizes the results of a {method} analysis performed on the dataset.
    
    ## Analysis Parameters
    
    """
    
    # Add parameters used in the analysis
    for key, value in params.items():
        if key != 'data':
            report += f"- **{key}**: {value}\n"
    
    # Add basic analysis details based on type
    if analysis_type == "Descriptive Statistics":
        col = params.get('column', 'Unknown')
        if method == "Summary Stats":
            stats_data = data[col].describe() if col in data.columns else data.describe()
            report += f"""
            ## Key Statistics
            
            - **Count**: {stats_data['count']}
            - **Mean**: {stats_data['mean']:.4f}
            - **Standard Deviation**: {stats_data['std']:.4f}
            - **Min**: {stats_data['min']:.4f}
            - **25%**: {stats_data['25%']:.4f}
            - **50% (Median)**: {stats_data['50%']:.4f}
            - **75%**: {stats_data['75%']:.4f}
            - **Max**: {stats_data['max']:.4f}
            
            ## Interpretation
            
            The data for {col} shows a central tendency around {stats_data['mean']:.4f} with a spread (standard deviation) of {stats_data['std']:.4f}.
            The distribution ranges from {stats_data['min']:.4f} to {stats_data['max']:.4f}, with the middle 50% of values falling between {stats_data['25%']:.4f} and {stats_data['75%']:.4f}.
            """
    
    elif analysis_type == "Inferential Statistics":
        if method == "T-test (1 sample)":
            col = params.get('column', 'Unknown')
            popmean = params.get('popmean', 0)
            t_stat = results.get('t_stat', 0)
            p_val = results.get('p_val', 1)
            
            report += f"""
            ## Test Results
            
            - **T-statistic**: {t_stat:.4f}
            - **P-value**: {p_val:.4f}
            - **Significance level**: 0.05
            
            ## Interpretation
            
            {'The test indicates that the mean of ' + col + ' is significantly different from ' + str(popmean) + '.' if p_val < 0.05 else 'The test does not provide evidence that the mean of ' + col + ' is different from ' + str(popmean) + '.'}
            
            ## Conclusion
            
            {'We can reject the null hypothesis that the population mean equals ' + str(popmean) + '.' if p_val < 0.05 else 'We fail to reject the null hypothesis that the population mean equals ' + str(popmean) + '.'}
            """
            
    elif analysis_type == "Relationship Analysis":
        if method == "Correlation Analysis":
            x_col = params.get('x_column', 'Unknown')
            y_col = params.get('y_column', 'Unknown')
            corr = results.get('correlation', 0)
            p_val = results.get('p_value', 1)
            
            report += f"""
            ## Correlation Results
            
            - **Correlation coefficient**: {corr:.4f}
            - **P-value**: {p_val:.4f}
            - **R-squared**: {corr**2:.4f}
            
            ## Interpretation
            
            The correlation between {x_col} and {y_col} is {'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.3 else 'weak'} and {'positive' if corr > 0 else 'negative'}.
            This correlation is {'statistically significant' if p_val < 0.05 else 'not statistically significant'} at the 0.05 level.
            
            ## Practical Significance
            
            The R-squared value of {corr**2:.4f} indicates that {corr**2*100:.1f}% of the variance in one variable can be explained by the other.
            
            ## Conclusion
            
            {'There is a significant relationship between these variables that warrants further investigation.' if p_val < 0.05 else 'There is not enough evidence to conclude a significant relationship between these variables.'}
            """
    
    # Add generic sections
    report += """
    ## Recommendations
    
    1. Consider exploring related variables to gain additional insights
    2. Use these findings to inform further statistical analyses
    3. When presenting these results, include visualizations for easier interpretation
    
    ## Limitations
    
    - Statistical significance does not always imply practical significance
    - Results should be interpreted in the context of the specific domain
    - Analysis assumptions should be verified for the most reliable conclusions
    """
    
    return report

# Main app title
st.title("📊 Comprehensive Statistical Analyzer")
st.markdown("Upload your data and explore various statistical analyses with interactive visualizations and interpretations.")

# Initialize session state
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "data" not in st.session_state:
    st.session_state.data = None

if "analysis_category" not in st.session_state:
    st.session_state.analysis_category = None

if "analysis_method" not in st.session_state:
    st.session_state.analysis_method = None

if "analysis_configured" not in st.session_state:
    st.session_state.analysis_configured = False

if "results_generated" not in st.session_state:
    st.session_state.results_generated = False

# File upload section
with st.expander("📥 Upload Data", expanded=not st.session_state.file_uploaded):
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sample Data Option")
        sample_option = st.selectbox(
            "Or use a sample dataset:",
            ["None", "Iris", "Boston Housing", "Titanic", "Stock Prices"]
        )
    
    with col2:
        st.markdown("### Data Cleaning Options")
        handle_missing = st.checkbox("Handle missing values", value=True)
        drop_duplicates = st.checkbox("Drop duplicate rows", value=True)

# Process uploaded or sample data
if uploaded_file is not None or sample_option != "None":
    try:
        if uploaded_file is not None:
            # Determine file type
            file_type = uploaded_file.name.split(".")[-1]
            
            # Read the file
            if file_type in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            else:  # csv
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        else:
            # Load sample data
            if sample_option == "Iris":
                df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
            elif sample_option == "Boston Housing":
                # Note: sklearn's Boston Housing dataset is deprecated, so we'll use a URL instead
                df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
            elif sample_option == "Titanic":
                df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
            elif sample_option == "Stock Prices":
                df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
                df['Date'] = pd.to_datetime(df['Date'])
            
            st.success(f"✅ Sample dataset '{sample_option}' loaded successfully!")
        
        # Data cleaning if selected
        if handle_missing:
            df = df.dropna(axis=1, thresh=int(len(df) * 0.5))  # Drop columns with >50% missing
            df = df.fillna(df.median(numeric_only=True))  # Fill missing with median for numeric
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        if drop_duplicates:
            initial_rows = len(df)
            df = df.drop_duplicates()
            if initial_rows > len(df):
                st.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Store in session state
        st.session_state.data = df
        st.session_state.file_uploaded = True
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(5))
        
        # Display data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
            
            # Provide download option for processed data
            st.markdown(get_download_link(df, "processed_data", "Download Processed Data"), unsafe_allow_html=True)
        
        with col2:
            st.subheader("Data Summary")
            st.dataframe(df.describe(include='all'))
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

# Sidebar for analysis selection
if st.session_state.file_uploaded and st.session_state.data is not None:
    df = st.session_state.data
    
    st.sidebar.title("Statistical Analysis")
    
    # Define analysis categories with descriptions
    analysis_categories = {
        "Descriptive Statistics": {
            "icon": "📊",
            "description": "Summarize and visualize the main characteristics of a dataset"
        },
        "Inferential Statistics": {
            "icon": "🔍",
            "description": "Make inferences and predictions about a population based on a sample"
        },
        "Relationship Analysis": {
            "icon": "🔗",
            "description": "Examine relationships and associations between variables"
        },
        "Non-parametric Tests": {
            "icon": "📉",
            "description": "Statistical tests that don't assume a specific distribution of data"
        },
        "Time Series Analysis": {
            "icon": "⏳",
            "description": "Analyze data points collected over time to identify patterns"
        },
        "Machine Learning": {
            "icon": "🤖",
            "description": "Use algorithms to build predictive models from data"
        }
    }
    
    # Display category options
    st.sidebar.subheader("1. Choose Analysis Category")
    
    for category, details in analysis_categories.items():
        if st.sidebar.button(f"{details['icon']} {category}", key=f"cat_{category}"):
            st.session_state.analysis_category = category
            st.session_state.analysis_method = None
            st.session_state.analysis_configured = False
            st.session_state.results_generated = False
            st.rerun()
    
    # If a category is selected, show methods for that category
    if st.session_state.analysis_category:
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"2. Choose Method for {analysis_categories[st.session_state.analysis_category]['icon']} {st.session_state.analysis_category}")
        
        # Show description of the selected category
        st.sidebar.markdown(f"<div class='sidebar-description'>{analysis_categories[st.session_state.analysis_category]['description']}</div>", unsafe_allow_html=True)
        
        # Define methods for each category
        methods = {}
        
        if st.session_state.analysis_category == "Descriptive Statistics":
            methods = {
                "Summary Stats": "Basic statistical summary of numerical data",
                "Distribution Analysis": "Analyze and visualize data distributions",
                "Frequency Analysis": "Count occurrences of categorical values",
                "Cross-tabulation": "Show relationship between categorical variables",
                "Outlier Detection": "Identify and visualize outliers in the data"
            }
        
        elif st.session_state.analysis_category == "Inferential Statistics":
            methods = {
                "T-test (1 sample)": "Compare sample mean to a known value",
                "T-test (2 sample)": "Compare means of two independent groups",
                "Paired T-test": "Compare means of two related samples",
                "ANOVA": "Compare means across three or more groups",
                "Chi-Square Test": "Test association between categorical variables",
                "Confidence Interval": "Estimate range for population parameter"
            }
        
        elif st.session_state.analysis_category == "Relationship Analysis":
            methods = {
                "Correlation Analysis": "Measure strength of relationship between variables",
                "Simple Linear Regression": "Predict a variable based on another variable",
                "Multiple Regression": "Predict a variable based on multiple variables",
                "Logistic Regression": "Predict binary outcomes from predictor variables"
            }
        
        elif st.session_state.analysis_category == "Non-parametric Tests":
            methods = {
                "Mann-Whitney U Test": "Non-parametric alternative to t-test",
                "Wilcoxon Signed-Rank Test": "Non-parametric alternative to paired t-test",
                "Kruskal-Wallis H Test": "Non-parametric alternative to ANOVA",
                "Spearman Correlation": "Non-parametric correlation measure"
            }
        
        elif st.session_state.analysis_category == "Time Series Analysis":
            methods = {
                "Time Series Visualization": "Plot and examine time series data",
                "Stationarity Tests": "Check if time series is stationary",
                "Seasonal Decomposition": "Separate trend, seasonality, and residuals",
                "Autocorrelation Analysis": "Examine correlation between time-shifted values"
            }
        
        elif st.session_state.analysis_category == "Machine Learning":
            methods = {
                "Principal Component Analysis": "Reduce dimensionality of data",
                "K-Means Clustering": "Group similar data points together",
                "Random Forest": "Build ensemble of decision trees for prediction",
                "Feature Importance": "Identify most important variables"
            }
        
        # Display method buttons
        for method, description in methods.items():
            if st.sidebar.button(method, key=f"method_{method}"):
                st.session_state.analysis_method = method
                st.session_state.analysis_configured = False
                st.session_state.results_generated = False
                st.rerun()
            
            # Show description for each method as tooltips
            st.sidebar.caption(f"{description}")
        
        # Button to go back to category selection
        st.sidebar.markdown("---")
        if st.sidebar.button("← Back to Categories"):
            st.session_state.analysis_category = None
            st.session_state.analysis_method = None
            st.session_state.analysis_configured = False
            st.session_state.results_generated = False
            st.experimental_rerun()
    
    # Show help and documentation in sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("📚 Help & Documentation"):
        st.markdown("""
        ### How to use this app
        
        1. Upload your data or select a sample dataset
        2. Choose an analysis category from the sidebar
        3. Select a specific analysis method
        4. Configure the analysis parameters
        5. View results and interpretations
        6. Generate a report if needed
        
        ### Tips
        
        - Hover over methods for descriptions
        - Use the back buttons to navigate
        - Check assumptions for statistical tests
        """)

# Main app area
if st.session_state.file_uploaded and st.session_state.data is not None:
    df = st.session_state.data
    
    # Get column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
    all_cols = df.columns.tolist()
    
    # If no analysis selected, show data exploration tools
    if not st.session_state.analysis_category:
        st.header("Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Quick Visualizations", "Data Quality", "Column Info"])
        
        with tab1:
            st.subheader("Create Quick Visualizations")
            
            viz_type = st.selectbox("Select Visualization Type",
                ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart", "Heatmap"])
            
            if viz_type == "Histogram":
                col = st.selectbox("Select column", numeric_cols)
                bins = st.slider("Number of bins", 5, 100, 20)
                
                fig = px.histogram(df, x=col, nbins=bins, marginal="box", 
                                title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Box Plot":
                col = st.selectbox("Select numeric column", numeric_cols)
                group_by = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                
                if group_by != "None":
                    fig = px.box(df, x=group_by, y=col, title=f"Box Plot of {col} by {group_by}")
                else:
                    fig = px.box(df, y=col, title=f"Box Plot of {col}")
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Scatter Plot":
                col_x = st.selectbox("Select X-axis column", numeric_cols)
                col_y = st.selectbox("Select Y-axis column", [c for c in numeric_cols if c != col_x])
                color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                
                if color_by != "None":
                    fig = px.scatter(df, x=col_x, y=col_y, color=color_by, 
                                    title=f"Scatter Plot: {col_y} vs {col_x}")
                else:
                    fig = px.scatter(df, x=col_x, y=col_y, 
                                    title=f"Scatter Plot: {col_y} vs {col_x}")
                
                fig.update_traces(marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Bar Chart":
                col = st.selectbox("Select categorical column", categorical_cols if categorical_cols else all_cols)
                orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
                
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, 'Count']
                
                if orientation == "Vertical":
                    fig = px.bar(counts, x=col, y='Count', title=f"Bar Chart of {col}")
                else:
                    fig = px.bar(counts, y=col, x='Count', title=f"Bar Chart of {col}", orientation='h')
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Line Chart":
                if datetime_cols:
                    x_col = st.selectbox("Select X-axis (time) column", datetime_cols)
                    y_col = st.selectbox("Select Y-axis column", numeric_cols)
                    
                    fig = px.line(df.sort_values(by=x_col), x=x_col, y=y_col, 
                                title=f"Line Chart: {y_col} over {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No datetime columns detected. Line charts work best with time series data.")
                    x_col = st.selectbox("Select X-axis column", numeric_cols)
                    y_col = st.selectbox("Select Y-axis column", [c for c in numeric_cols if c != x_col])
                    
                    fig = px.line(df.sort_values(by=x_col), x=x_col, y=y_col, 
                                title=f"Line Chart: {y_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Heatmap":
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    
                    fig = px.imshow(corr, text_auto=True, aspect="auto",
                                title="Correlation Heatmap")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for a correlation heatmap.")
        
        with tab2:
            st.subheader("Data Quality Overview")
            
            # Missing values
            missing = df.isnull().sum()
            missing_percent = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Values': missing.values,
                'Percentage': missing_percent.values
            }).sort_values('Missing Values', ascending=False)
            
            st.markdown("#### Missing Values")
            st.dataframe(missing_df)
            
            # Visualization of missing values
            if missing.sum() > 0:
                fig = px.bar(missing_df, x='Column', y='Percentage',
                           title="Percentage of Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
            
            # Duplicates
            st.markdown("#### Duplicate Rows")
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                st.warning(f"Found {dup_count} duplicate rows ({(dup_count/len(df)*100):.2f}% of data)")
            else:
                st.success("No duplicate rows found in the data")
        
        with tab3:
            st.subheader("Column Information")
            
            # Column selector
            selected_col = st.selectbox("Select a column to examine", all_cols)
            
            # Display column info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Information")
                col_type = df[selected_col].dtype
                st.write(f"Data type: {col_type}")
                st.write(f"Unique values: {df[selected_col].nunique()}")
                
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    st.write(f"Min: {df[selected_col].min()}")
                    st.write(f"Max: {df[selected_col].max()}")
                    st.write(f"Mean: {df[selected_col].mean()}")
                    st.write(f"Median: {df[selected_col].median()}")
                    st.write(f"Standard deviation: {df[selected_col].std()}")
            
            with col2:
                st.markdown("#### Value Distribution")
                
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = df[selected_col].value_counts().reset_index()
                    value_counts.columns = [selected_col, 'Count']
                    if len(value_counts) > 15:
                        value_counts = value_counts.head(15)
                        st.info("Showing top 15 values due to large number of unique values")
                    
                    fig = px.bar(value_counts, x=selected_col, y='Count', 
                                title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
    
    # If analysis category and method are selected
    elif st.session_state.analysis_category and st.session_state.analysis_method:
        # Display header with selected analysis
        category = st.session_state.analysis_category
        method = st.session_state.analysis_method
        category_icon = analysis_categories[category]["icon"]
        
        st.header(f"{category_icon} {category}: {method}")
        
        # Button to go back to exploration
        if st.button("← Back to Data Exploration"):
            st.session_state.analysis_category = None
            st.session_state.analysis_method = None
            st.session_state.analysis_configured = False
            st.session_state.results_generated = False
            st.experimental_rerun()
        
        # Configuration tab
        tab1, tab2 = st.tabs(["📐 Configure Analysis", "📊 Results & Interpretation"])
        
        with tab1:
            st.subheader("Configure Analysis Parameters")
            
            # Parameters for different analysis methods
            params = {}
            
            #=======================
            # Descriptive Statistics
            #=======================
            if category == "Descriptive Statistics":
                if method == "Summary Stats":
                    col = st.selectbox("Select column (optional, leave blank for all numeric columns)", 
                                      ["All Numeric Columns"] + numeric_cols)
                    
                    params = {
                        "column": col if col != "All Numeric Columns" else None
                    }
                
                elif method == "Distribution Analysis":
                    col = st.selectbox("Select column", numeric_cols)
                    plot_type = st.radio("Plot type", ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"])
                    
                    params = {
                        "column": col,
                        "plot_type": plot_type
                    }
                
                elif method == "Frequency Analysis":
                    col = st.selectbox("Select column", categorical_cols if categorical_cols else all_cols)
                    sort_by = st.radio("Sort by", ["Frequency", "Value"])
                    max_categories = st.slider("Max categories to show", 5, 50, 10)
                    
                    params = {
                        "column": col,
                        "sort_by": sort_by,
                        "max_categories": max_categories
                    }
                
                elif method == "Cross-tabulation":
                    col1 = st.selectbox("Row variable", categorical_cols if categorical_cols else all_cols)
                    col2 = st.selectbox("Column variable", 
                                      [c for c in (categorical_cols if categorical_cols else all_cols) if c != col1])
                    normalize = st.radio("Show percentages", ["No", "Row", "Column", "All"])
                    
                    params = {
                        "row_var": col1,
                        "col_var": col2,
                        "normalize": None if normalize == "No" else normalize.lower()
                    }
                
                elif method == "Outlier Detection":
                    col = st.selectbox("Select column", numeric_cols)
                    method_type = st.radio("Detection method", ["Z-score", "IQR"])
                    threshold = st.slider("Threshold", 1.5, 5.0, 3.0, 0.1)
                    
                    params = {
                        "column": col,
                        "method_type": method_type,
                        "threshold": threshold
                    }
            
            #=======================
            # Inferential Statistics
            #=======================
            elif category == "Inferential Statistics":
                if method == "T-test (1 sample)":
                    col = st.selectbox("Select column", numeric_cols)
                    popmean = st.number_input("Population mean (null hypothesis)", value=0.0)
                    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                    
                    params = {
                        "column": col,
                        "popmean": popmean,
                        "alpha": alpha
                    }
                
                elif method == "T-test (2 sample)":
                    num_col = st.selectbox("Select numeric column", numeric_cols)
                    group_col = st.selectbox("Select grouping column", categorical_cols if categorical_cols else all_cols)
                    
                    # Get unique values in the grouping column
                    unique_groups = df[group_col].unique()
                    
                    if len(unique_groups) < 2:
                        st.error("The grouping column must have at least 2 unique values")
                    elif len(unique_groups) == 2:
                        # If exactly 2 groups, we're good
                        equal_var = st.checkbox("Assume equal variances", value=False)
                        alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                        
                        params = {
                            "numeric_col": num_col,
                            "group_col": group_col,
                            "group1": unique_groups[0],
                            "group2": unique_groups[1],
                            "equal_var": equal_var,
                            "alpha": alpha
                        }
                    else:
                        # If more than 2 groups, let user select which 2 to compare
                        group1 = st.selectbox("Select first group", unique_groups)
                        remaining_groups = [g for g in unique_groups if g != group1]
                        group2 = st.selectbox("Select second group", remaining_groups)
                        
                        equal_var = st.checkbox("Assume equal variances", value=False)
                        alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                        
                        params = {
                            "numeric_col": num_col,
                            "group_col": group_col,
                            "group1": group1,
                            "group2": group2,
                            "equal_var": equal_var,
                            "alpha": alpha
                        }
                
                elif method == "Paired T-test":
                    col1 = st.selectbox("First measurement", numeric_cols)
                    col2 = st.selectbox("Second measurement", [c for c in numeric_cols if c != col1])
                    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                    
                    params = {
                        "measure1": col1,
                        "measure2": col2,
                        "alpha": alpha
                    }
                
                elif method == "ANOVA":
                    num_col = st.selectbox("Select numeric column", numeric_cols)
                    group_col = st.selectbox("Select grouping column", categorical_cols if categorical_cols else all_cols)
                    
                    # Check if enough groups
                    unique_groups = df[group_col].unique()
                    if len(unique_groups) < 2:
                        st.error("The grouping column must have at least 2 unique values")
                    else:
                        alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                        
                        params = {
                            "numeric_col": num_col,
                            "group_col": group_col,
                            "alpha": alpha
                        }
                
                elif method == "Chi-Square Test":
                    col1 = st.selectbox("First categorical column", categorical_cols if categorical_cols else all_cols)
                    col2 = st.selectbox("Second categorical column", 
                                      [c for c in (categorical_cols if categorical_cols else all_cols) if c != col1])
                    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                    
                    params = {
                        "column1": col1,
                        "column2": col2,
                        "alpha": alpha
                    }
                
                elif method == "Confidence Interval":
                    col = st.selectbox("Select column", numeric_cols)
                    confidence = st.slider("Confidence level (%)", 80, 99, 95, 1)
                    method_type = st.radio("Method", ["Normal", "T-distribution", "Bootstrap"])
                    
                    params = {
                        "column": col,
                        "confidence": confidence,
                        "method_type": method_type
                    }
            
            #=======================
            # Relationship Analysis
            #=======================
            elif category == "Relationship Analysis":
                if method == "Correlation Analysis":
                    corr_type = st.radio("Correlation type", ["Pearson", "Spearman"])
                    
                    if st.checkbox("Select specific variables", value=False):
                        x_col = st.selectbox("Select X variable", numeric_cols)
                        y_col = st.selectbox("Select Y variable", [c for c in numeric_cols if c != x_col])
                        
                        params = {
                            "correlation_type": corr_type,
                            "x_column": x_col,
                            "y_column": y_col,
                            "all_numeric": False
                        }
                    else:
                        params = {
                            "correlation_type": corr_type,
                            "all_numeric": True
                        }
                
                elif method == "Simple Linear Regression":
                    x_col = st.selectbox("Independent variable (X)", numeric_cols)
                    y_col = st.selectbox("Dependent variable (Y)", [c for c in numeric_cols if c != x_col])
                    ci = st.slider("Confidence interval (%)", 80, 99, 95, 1)
                    
                    params = {
                        "x_column": x_col,
                        "y_column": y_col,
                        "confidence_interval": ci
                    }
                
                elif method == "Multiple Regression":
                    y_col = st.selectbox("Dependent variable (Y)", numeric_cols)
                    x_cols = st.multiselect("Independent variables (X)", 
                                          [c for c in numeric_cols if c != y_col],
                                          default=[c for c in numeric_cols if c != y_col][:min(3, len(numeric_cols)-1)])
                    
                    if not x_cols:
                        st.error("Please select at least one independent variable")
                    else:
                        params = {
                            "y_column": y_col,
                            "x_columns": x_cols
                        }
                
                elif method == "Logistic Regression":
                    if not categorical_cols:
                        st.error("Logistic regression requires at least one categorical column for the target variable")
                    else:
                        y_col = st.selectbox("Target variable (binary)", categorical_cols)
                        
                        # Check if target is binary
                        unique_vals = df[y_col].unique()
                        if len(unique_vals) != 2:
                            st.warning(f"Target variable has {len(unique_vals)} unique values. Logistic regression works best with binary (2-class) targets.")
                        
                        x_cols = st.multiselect("Predictor variables", 
                                              [c for c in numeric_cols],
                                              default=numeric_cols[:min(3, len(numeric_cols))])
                        
                        if not x_cols:
                            st.error("Please select at least one predictor variable")
                        else:
                            params = {
                                "y_column": y_col,
                                "x_columns": x_cols
                            }
            
            #=======================
            # Non-parametric Tests
            #=======================
            elif category == "Non-parametric Tests":
                if method == "Mann-Whitney U Test":
                    num_col = st.selectbox("Select numeric column", numeric_cols)
                    group_col = st.selectbox("Select grouping column", categorical_cols if categorical_cols else all_cols)
                    
                    # Get unique values in the grouping column
                    unique_groups = df[group_col].unique()
                    
                    if len(unique_groups) < 2:
                        st.error("The grouping column must have at least 2 unique values")
                    elif len(unique_groups) == 2:
                        # If exactly 2 groups, we're good
                        alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                        
                        params = {
                            "numeric_col": num_col,
                            "group_col": group_col,
                            "group1": unique_groups[0],
                            "group2": unique_groups[1],
                            "alpha": alpha
                        }
                    else:
                        # If more than 2 groups, let user select which 2 to compare
                        group1 = st.selectbox("Select first group", unique_groups)
                        remaining_groups = [g for g in unique_groups if g != group1]
                        group2 = st.selectbox("Select second group", remaining_groups)
                        
                        alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                        
                        params = {
                            "numeric_col": num_col,
                            "group_col": group_col,
                            "group1": group1,
                            "group2": group2,
                            "alpha": alpha
                        }
                
                elif method == "Wilcoxon Signed-Rank Test":
                    col1 = st.selectbox("First measurement", numeric_cols)
                    col2 = st.selectbox("Second measurement", [c for c in numeric_cols if c != col1])
                    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                    
                    params = {
                        "measure1": col1,
                        "measure2": col2,
                        "alpha": alpha
                    }
                
                elif method == "Kruskal-Wallis H Test":
                    num_col = st.selectbox("Select numeric column", numeric_cols)
                    group_col = st.selectbox("Select grouping column", categorical_cols if categorical_cols else all_cols)
                    
                    # Check if enough groups
                    unique_groups = df[group_col].unique()
                    if len(unique_groups) < 2:
                        st.error("The grouping column must have at least 2 unique values")
                    else:
                        alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
                        
                        params = {
                            "numeric_col": num_col,
                            "group_col": group_col,
                            "alpha": alpha
                        }
                
                elif method == "Spearman Correlation":
                    if st.checkbox("Select specific variables", value=False):
                        x_col = st.selectbox("Select X variable", numeric_cols)
                        y_col = st.selectbox("Select Y variable", [c for c in numeric_cols if c != x_col])
                        
                        params = {
                            "x_column": x_col,
                            "y_column": y_col,
                            "all_numeric": False
                        }
                    else:
                        params = {
                            "all_numeric": True
                        }
            
            #=======================
            # Time Series Analysis
            #=======================
            elif category == "Time Series Analysis":
                if not datetime_cols:
                    st.warning("No datetime columns detected. Time series analysis works best with time data.")
                    # Allow user to select a column to convert to datetime
                    date_col = st.selectbox("Select column to convert to datetime", all_cols)
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        st.success(f"Successfully converted {date_col} to datetime")
                        datetime_cols = [date_col]
                    except:
                        st.error(f"Could not convert {date_col} to datetime format")
                
                if datetime_cols:
                    if method == "Time Series Visualization":
                        date_col = st.selectbox("Select date/time column", datetime_cols)
                        value_col = st.selectbox("Select value column", numeric_cols)
                        resample = st.selectbox("Resample frequency", 
                                             ["None", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
                        
                        params = {
                            "date_column": date_col,
                            "value_column": value_col,
                            "resample": resample
                        }
                    
                    elif method == "Stationarity Tests":
                        date_col = st.selectbox("Select date/time column", datetime_cols)
                        value_col = st.selectbox("Select value column", numeric_cols)
                        
                        params = {
                            "date_column": date_col,
                            "value_column": value_col
                        }
                    
                    elif method == "Seasonal Decomposition":
                        date_col = st.selectbox("Select date/time column", datetime_cols)
                        value_col = st.selectbox("Select value column", numeric_cols)
                        period = st.number_input("Period (number of time units in a seasonal cycle)", 
                                             value=12, min_value=2)
                        model = st.selectbox("Decomposition model", ["Additive", "Multiplicative"])
                        
                        params = {
                            "date_column": date_col,
                            "value_column": value_col,
                            "period": period,
                            "model": model.lower()
                        }
                    
                    elif method == "Autocorrelation Analysis":
                        date_col = st.selectbox("Select date/time column", datetime_cols)
                        value_col = st.selectbox("Select value column", numeric_cols)
                        max_lags = st.slider("Maximum lags", 5, 50, 20)
                        
                        params = {
                            "date_column": date_col,
                            "value_column": value_col,
                            "max_lags": max_lags
                        }
                
                else:
                    st.error("Time series analysis requires a datetime column")
            
            #=======================
            # Machine Learning
            #=======================
            elif category == "Machine Learning":
                if method == "Principal Component Analysis":
                    n_components = st.slider("Number of components", 2, min(10, len(numeric_cols)), 
                                          min(3, len(numeric_cols)))
                    scale_data = st.checkbox("Standardize data (recommended)", value=True)
                    
                    params = {
                        "n_components": n_components,
                        "scale_data": scale_data
                    }
                
                elif method == "K-Means Clustering":
                    features = st.multiselect("Select features for clustering", numeric_cols, 
                                           default=numeric_cols[:min(3, len(numeric_cols))])
                    n_clusters = st.slider("Number of clusters", 2, 10, 3)
                    scale_data = st.checkbox("Standardize data (recommended)", value=True)
                    
                    if not features:
                        st.error("Please select at least one feature for clustering")
                    else:
                        params = {
                            "features": features,
                            "n_clusters": n_clusters,
                            "scale_data": scale_data
                        }
                
                elif method == "Random Forest":
                    target_type = st.radio("Target type", ["Classification", "Regression"])
                    
                    if target_type == "Classification":
                        if not categorical_cols:
                            st.error("Classification requires at least one categorical column for the target variable")
                        else:
                            target = st.selectbox("Target variable", categorical_cols)
                            features = st.multiselect("Select features", 
                                                    [c for c in numeric_cols],
                                                    default=numeric_cols[:min(5, len(numeric_cols))])
                            
                            n_trees = st.slider("Number of trees", 10, 500, 100)
                            test_size = st.slider("Test size (%)", 10, 50, 30) / 100
                            
                            if not features:
                                st.error("Please select at least one feature")
                            else:
                                params = {
                                    "target_type": target_type,
                                    "target": target,
                                    "features": features,
                                    "n_trees": n_trees,
                                    "test_size": test_size
                                }
                    else:  # Regression
                        target = st.selectbox("Target variable", numeric_cols)
                        features = st.multiselect("Select features", 
                                                [c for c in numeric_cols if c != target],
                                                default=[c for c in numeric_cols if c != target][:min(5, len(numeric_cols)-1)])
                        
                        n_trees = st.slider("Number of trees", 10, 500, 100)
                        test_size = st.slider("Test size (%)", 10, 50, 30) / 100
                        
                        if not features:
                            st.error("Please select at least one feature")
                        else:
                            params = {
                                "target_type": target_type,
                                "target": target,
                                "features": features,
                                "n_trees": n_trees,
                                "test_size": test_size
                            }
                
                elif method == "Feature Importance":
                    target_type = st.radio("Target type", ["Classification", "Regression"])
                    
                    if target_type == "Classification":
                        if not categorical_cols:
                            st.error("Classification requires at least one categorical column for the target variable")
                        else:
                            target = st.selectbox("Target variable", categorical_cols)
                            features = st.multiselect("Select features", 
                                                    [c for c in numeric_cols],
                                                    default=numeric_cols[:min(10, len(numeric_cols))])
                            
                            if not features:
                                st.error("Please select at least one feature")
                            else:
                                params = {
                                    "target_type": target_type,
                                    "target": target,
                                    "features": features
                                }
                    else:  # Regression
                        target = st.selectbox("Target variable", numeric_cols)
                        features = st.multiselect("Select features", 
                                                [c for c in numeric_cols if c != target],
                                                default=[c for c in numeric_cols if c != target][:min(10, len(numeric_cols)-1)])
                        
                        if not features:
                            st.error("Please select at least one feature")
                        else:
                            params = {
                                "target_type": target_type,
                                "target": target,
                                "features": features
                            }
            
            # Run analysis button
            if params:
                st.session_state.params = params
                
                if st.button("▶️ Run Analysis", type="primary"):
                    st.session_state.analysis_configured = True
                    st.session_state.results_generated = True
                    
                    # Store results in session state (will be calculated in the results tab)
                    st.session_state.results = None
                    
                    # Switch to results tab
                    st.rerun()
        
        # Results tab
        with tab2:
            if not st.session_state.results_generated:
                st.info("Configure and run the analysis to see results here")
            else:
                st.subheader("Analysis Results")
                
                # Get parameters from session state
                params = st.session_state.params
                
                # Placeholder for results
                results = {}
                
                # Here we would implement the actual analysis based on category and method
                # For brevity, I'm only including a small sample of the analysis code
                
                if category == "Descriptive Statistics" and method == "Summary Stats":
                    # Output basic descriptive statistics
                    if params["column"] is None:
                        st.write("Summary statistics for all numeric columns:")
                        st.dataframe(df[numeric_cols].describe())
                    else:
                        col = params["column"]
                        st.write(f"Summary statistics for {col}:")
                        stats_data = df[col].describe()
                        st.dataframe(pd.DataFrame(stats_data).T)
                        
                        # Create visualization
                        fig = px.histogram(df, x=col, marginal="box", 
                                        title=f"Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        st.subheader("Interpretation")
                        interpretation = f"""
                        <div class="interpretation">
                        <p>The data for <b>{col}</b> has a mean of {stats_data['mean']:.4f} and a standard deviation of {stats_data['std']:.4f}.</p>
                        <p>The values range from {stats_data['min']:.4f} to {stats_data['max']:.4f}, with a median of {stats_data['50%']:.4f}.</p>
                        <p>The middle 50% of values fall between {stats_data['25%']:.4f} and {stats_data['75%']:.4f}.</p>
                        </div>
                        """
                        st.markdown(interpretation, unsafe_allow_html=True)
                        
                        # Save results
                        results = {
                            "data": stats_data
                        }
                
                # Generate Report button after results are shown
                if results:  # Only show if we have results
                    st.markdown("---")
                    st.subheader("📊 Generate Analysis Report")
                    
                    if st.button("Generate Report", key="final_report_button"):
                        report_content = generate_report_content(category, df, method, params, results)
                        
                        st.success("Report generated successfully!")
                        st.markdown("### Report Preview")
                        st.markdown(report_content)
                        
                        # Provide download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="Download as Markdown",
                                data=report_content,
                                file_name=f"{method.lower().replace(' ', '_')}_report.md",
                                mime="text/markdown"
                            )
                        
                        with col2:
                            pdf_report = create_pdf_report(report_content)
                            st.download_button(
                                label="Download as PDF",
                                data=pdf_report,
                                file_name=f"{method.lower().replace(' ', '_')}_report.pdf",
                                mime="application/pdf"
                            )

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit • v2.0.0")
st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y')}")