import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.api as sm
import io
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="Comprehensive Statistical Analyzer", layout="wide")

# Initialize session state variables
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# Function to query LLM for analysis
def query_llm(prompt, api_key, model="gpt-3.5-turbo"):
    if not api_key:
        return "Please enter an OpenAI API key in the sidebar settings to use the LLM analysis feature."
    
    client = OpenAI(api_key=api_key)
    try:
        system_message = """You are an expert data analysis assistant that helps users understand their data and statistical results.

GUIDELINES FOR YOUR RESPONSES:
1. Provide thoughtful and accurate analysis based on the data provided
2. Use clear, concise language that non-technical users can understand
3. Highlight key findings and what they mean in practical terms
4. Explain any statistical significance found in the results
5. Mention limitations or caveats in the analysis when relevant
6. Format your explanation with markdown headers and bullet points when helpful
7. Provide actionable insights based on the data or statistical analysis
8. When appropriate, suggest additional analyses that might yield valuable insights"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying OpenAI API: {str(e)}"

# Function to set the chat question (for example buttons)
def set_question(question):
    st.session_state.user_question = question

# Define tabs for the application
tab1, tab2 = st.tabs(["📊 Statistical Analysis", "💬 Chat with Data"])

# Sidebar for settings and file upload
with st.sidebar:
    st.header("Settings")
    
    # API key settings
    api_key_expander = st.expander("🤖 LLM Settings", expanded=False)
    with api_key_expander:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        openai_model = st.selectbox(
            "Select OpenAI model:",
            ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o")
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])

# Process uploaded file
df = None
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        
        st.sidebar.success(f"{file_type.upper()} file uploaded successfully!")
        
        # Display file preview in sidebar
        with st.sidebar.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(3), use_container_width=True)
            
        # Add columns count and data types summary to the sidebar
        with st.sidebar.expander("📊 Data Overview", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            
            # Data types summary
            dtypes_count = df.dtypes.value_counts().reset_index()
            dtypes_count.columns = ['Data Type', 'Count']
            st.dataframe(dtypes_count, use_container_width=True)
            
            # Missing values
            missing_vals = df.isnull().sum().sum()
            if missing_vals > 0:
                st.warning(f"⚠️ Dataset contains {missing_vals:,} missing values")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {str(e)}")
        df = None

# Main application logic
if df is not None:
    # Variables that will be used across tabs
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Analysis type selection in sidebar (only for Statistical Analysis tab)
    analysis_type = st.sidebar.selectbox("Choose Analysis Type", [
        "Descriptive Statistics",
        "Inferential Statistics",
        "Non-parametric Tests",
        "Time Series Analysis",
        "Advanced / Multivariate Analysis"
    ])
    
    # Statistical Analysis Tab Content
    with tab1:
        st.title("📊 Comprehensive Statistical Analyzer")
        st.dataframe(df.head(), use_container_width=True)

        if analysis_type == "Descriptive Statistics":
            method = st.sidebar.selectbox("Choose Method", ["Summary Stats", "Frequency Table", "Cross-tabulation", "Boxplot", "Histogram"])

            if method == "Summary Stats":
                st.subheader("Descriptive Summary")
                st.dataframe(df.describe(include='all'))
                results = df.describe(include='all')

            elif method == "Frequency Table":
                col = st.sidebar.selectbox("Select column", all_cols)
                st.subheader(f"Frequency Table for {col}")
                freq_table = df[col].value_counts().reset_index().rename(columns={'index': col, col: 'Frequency'})
                st.dataframe(freq_table)
                results = freq_table

            elif method == "Cross-tabulation":
                col1 = st.sidebar.selectbox("Column 1", all_cols)
                col2 = st.sidebar.selectbox("Column 2", all_cols)
                st.subheader(f"Cross-tab: {col1} vs {col2}")
                cross_tab = pd.crosstab(df[col1], df[col2])
                st.dataframe(cross_tab)
                results = cross_tab

            elif method == "Boxplot":
                col = st.sidebar.selectbox("Numeric column", numeric_cols)
                st.subheader(f"Boxplot of {col}")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)
                results = f"Boxplot statistics for {col}: Min={df[col].min()}, Q1={df[col].quantile(0.25)}, Median={df[col].median()}, Q3={df[col].quantile(0.75)}, Max={df[col].max()}"

            elif method == "Histogram":
                col = st.sidebar.selectbox("Numeric column", numeric_cols)
                st.subheader(f"Histogram of {col}")
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)
                results = f"Histogram statistics for {col}: Mean={df[col].mean()}, Std={df[col].std()}, Min={df[col].min()}, Max={df[col].max()}"

        elif analysis_type == "Inferential Statistics":
            method = st.sidebar.selectbox("Choose Test", [
                "T-test (1 sample)",
                "T-test (2 sample)",
                "Paired T-test",
                "ANOVA",
                "Chi-Square Test",
                "Correlation",
                "Confidence Interval"
            ])

            if method == "T-test (1 sample)":
                col = st.sidebar.selectbox("Select numeric column", numeric_cols)
                popmean = st.sidebar.number_input("Population mean", value=0.0)
                t_stat, p_val = stats.ttest_1samp(df[col].dropna(), popmean)
                results = f"T-test (1 sample) for {col}: T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}"
                st.write(results)

            elif method == "T-test (2 sample)":
                col = st.sidebar.selectbox("Select numeric column", numeric_cols)
                group_col = st.sidebar.selectbox("Group by column (2 groups)", all_cols)
                groups = df[group_col].dropna().unique()
                if len(groups) == 2:
                    group1 = df[df[group_col] == groups[0]][col].dropna()
                    group2 = df[df[group_col] == groups[1]][col].dropna()
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    results = f"T-test (2 sample) for {col} grouped by {group_col}: T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}"
                    st.write(results)
                else:
                    st.warning("Column must have exactly 2 unique values")
                    results = "Error: Column must have exactly 2 unique values"

            elif method == "Paired T-test":
                col1 = st.sidebar.selectbox("First sample", numeric_cols)
                col2 = st.sidebar.selectbox("Second sample", numeric_cols)
                t_stat, p_val = stats.ttest_rel(df[col1].dropna(), df[col2].dropna())
                results = f"Paired T-test for {col1} and {col2}: T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}"
                st.write(results)

            elif method == "ANOVA":
                col = st.sidebar.selectbox("Numeric column", numeric_cols)
                group_col = st.sidebar.selectbox("Group by column", all_cols)
                groups = [group[col].dropna() for name, group in df.groupby(group_col)]
                f_stat, p_val = stats.f_oneway(*groups)
                results = f"ANOVA for {col} grouped by {group_col}: F-statistic: {f_stat:.4f}, P-value: {p_val:.4f}"
                st.write(results)

            elif method == "Chi-Square Test":
                col1 = st.sidebar.selectbox("Column 1", all_cols)
                col2 = st.sidebar.selectbox("Column 2", all_cols)
                table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, expected = stats.chi2_contingency(table)
                results = f"Chi-Square Test for {col1} and {col2}: Chi2: {chi2:.4f}, P-value: {p:.4f}, DOF: {dof}"
                st.write(results)

            elif method == "Correlation":
                method_type = st.sidebar.selectbox("Method", ["pearson", "spearman"])
                cols = st.sidebar.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:2])
                if len(cols) >= 2:
                    corr = df[cols].corr(method=method_type)
                    st.write(f"{method_type.title()} Correlation Matrix")
                    st.dataframe(corr)
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                    results = f"{method_type.title()} Correlation Matrix:\n{corr.to_string()}"
                else:
                    results = "Error: Need at least 2 columns for correlation analysis"

            elif method == "Confidence Interval":
                col = st.sidebar.selectbox("Select numeric column", numeric_cols)
                ci_level = st.sidebar.slider("Confidence level", 80, 99, 95)
                data = df[col].dropna()
                mean = data.mean()
                sem = stats.sem(data)
                margin = sem * stats.t.ppf((1 + ci_level / 100) / 2., len(data)-1)
                results = f"{ci_level}% Confidence Interval for {col}: ({mean - margin:.4f}, {mean + margin:.4f})"
                st.write(results)

        elif analysis_type == "Non-parametric Tests":
            method = st.sidebar.selectbox("Choose Method", ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Kruskal-Wallis H Test"])

            if method == "Mann-Whitney U Test":
                col = st.sidebar.selectbox("Numeric column", numeric_cols)
                group_col = st.sidebar.selectbox("Group column", all_cols)
                groups = df[group_col].dropna().unique()
                if len(groups) == 2:
                    data1 = df[df[group_col] == groups[0]][col].dropna()
                    data2 = df[df[group_col] == groups[1]][col].dropna()
                    u_stat, p_val = stats.mannwhitneyu(data1, data2)
                    results = f"Mann-Whitney U Test for {col} grouped by {group_col}: U-statistic: {u_stat:.4f}, P-value: {p_val:.4f}"
                    st.write(results)
                else:
                    results = "Error: Need exactly 2 groups for Mann-Whitney U Test"

            elif method == "Wilcoxon Signed-Rank Test":
                col1 = st.sidebar.selectbox("First sample", numeric_cols)
                col2 = st.sidebar.selectbox("Second sample", numeric_cols)
                w_stat, p_val = stats.wilcoxon(df[col1].dropna(), df[col2].dropna())
                results = f"Wilcoxon Signed-Rank Test for {col1} and {col2}: W-statistic: {w_stat:.4f}, P-value: {p_val:.4f}"
                st.write(results)

            elif method == "Kruskal-Wallis H Test":
                col = st.sidebar.selectbox("Numeric column", numeric_cols)
                group_col = st.sidebar.selectbox("Group column", all_cols)
                groups = [group[col].dropna() for name, group in df.groupby(group_col)]
                h_stat, p_val = stats.kruskal(*groups)
                results = f"Kruskal-Wallis H Test for {col} grouped by {group_col}: H-statistic: {h_stat:.4f}, P-value: {p_val:.4f}"
                st.write(results)

        elif analysis_type == "Time Series Analysis":
            col = st.sidebar.selectbox("Select time series column", numeric_cols)
            if col:
                st.line_chart(df[col])
                result = adfuller(df[col].dropna())
                results = f"Time Series Analysis for {col}: ADF Statistic: {result[0]:.4f}, P-value: {result[1]:.4f}"
                st.write(results)
                st.write("Time Series is likely stationary" if result[1] < 0.05 else "Non-stationary")
                results += ", Status: " + ("Likely stationary" if result[1] < 0.05 else "Non-stationary")

        elif analysis_type == "Advanced / Multivariate Analysis":
            method = st.sidebar.selectbox("Choose Method", ["Linear Regression", "Logistic Regression", "PCA", "K-Means Clustering", "Random Forest"])

            if method == "Linear Regression":
                target = st.sidebar.selectbox("Target variable", numeric_cols)
                features = st.sidebar.multiselect("Feature variables", [col for col in numeric_cols if col != target])
                if features:
                    X = df[features].dropna()
                    y = df[target].loc[X.index]
                    model = LinearRegression().fit(X, y)
                    coefficients = dict(zip(features, model.coef_))
                    st.write("Coefficients:", coefficients)
                    st.write(f"Intercept: {model.intercept_:.4f}, R-squared: {model.score(X, y):.4f}")
                    results = f"Linear Regression for {target}: Coefficients: {coefficients}, Intercept: {model.intercept_:.4f}, R-squared: {model.score(X, y):.4f}"
                else:
                    results = "Error: No features selected for Linear Regression"

            elif method == "Logistic Regression":
                target = st.sidebar.selectbox("Target variable (binary)", all_cols)
                features = st.sidebar.multiselect("Feature variables", numeric_cols)
                if features:
                    X = df[features].dropna()
                    y = df[target].loc[X.index]
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X, y)
                    coefficients = dict(zip(features, model.coef_[0]))
                    st.write("Coefficients:", coefficients)
                    st.write(f"Intercept: {model.intercept_[0]:.4f}")
                    results = f"Logistic Regression for {target}: Coefficients: {coefficients}, Intercept: {model.intercept_[0]:.4f}"
                else:
                    results = "Error: No features selected for Logistic Regression"

            elif method == "PCA":
                n_components = st.sidebar.slider("Number of components", 1, len(numeric_cols), 2)
                X = df[numeric_cols].dropna()
                X_scaled = StandardScaler().fit_transform(X)
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(X_scaled)
                st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
                st.dataframe(pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)]))
                results = f"PCA with {n_components} components: Explained Variance Ratio: {pca.explained_variance_ratio_}"

            elif method == "K-Means Clustering":
                X = df[numeric_cols].dropna()
                k = st.sidebar.slider("Number of clusters", 2, 10, 3)
                kmeans = KMeans(n_clusters=k)
                X_scaled = StandardScaler().fit_transform(X)
                labels = kmeans.fit_predict(X_scaled)
                df_with_clusters = df.copy()
                df_with_clusters['Cluster'] = labels
                st.write("Cluster Centers:", kmeans.cluster_centers_)
                st.dataframe(df_with_clusters.head())
                results = f"K-Means Clustering with {k} clusters: Created clusters and assigned labels to data"

            elif method == "Random Forest":
                target = st.sidebar.selectbox("Target variable", all_cols)
                features = st.sidebar.multiselect("Feature variables", numeric_cols)
                if features:
                    X = df[features].dropna()
                    y = df[target].loc[X.index]
                    model = RandomForestClassifier()
                    model.fit(X, y)
                    importances = model.feature_importances_
                    feature_importances = dict(zip(features, importances))
                    st.write("Feature Importances:", feature_importances)
                    results = f"Random Forest for {target}: Feature Importances: {feature_importances}"
                else:
                    results = "Error: No features selected for Random Forest"

        # Store results in session state
        st.session_state.analysis_results = results
        
        # Generate LLM Report Button
        st.markdown("---")
        st.subheader("🤖 AI Analysis Report")
        
        if st.button("Generate AI Analysis Report"):
            if not api_key:
                st.error("Please enter an OpenAI API key in the sidebar to use the AI report feature.")
            else:
                with st.spinner("Generating analysis report..."):
                    # Capture DataFrame information 
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    df_info = buffer.getvalue()
                    
                    # Create prompt with dataframe info and analysis results
                    prompt = f"""
                    I've performed a statistical analysis and need a detailed explanation of the results.
                    
                    Dataset Information:
                    ```
                    DataFrame head:
                    {df.head().to_string()}
                    
                    DataFrame description:
                    {df.describe().to_string()}
                    
                    DataFrame info:
                    {df_info}
                    ```
                    
                    Analysis Type: {analysis_type}
                    Method: {method}
                    
                    Results:
                    ```
                    {st.session_state.analysis_results}
                    ```
                    
                    Please provide:
                    1. A concise summary of what this data represents
                    2. A detailed explanation of the statistical results in simple terms
                    3. What these results mean in practical terms
                    4. Any limitations or caveats to be aware of
                    5. Potential next steps or recommendations based on these findings
                    
                    Format your response with clear headers and bullet points when appropriate.
                    """
                    
                    # Query LLM and display results
                    llm_response = query_llm(prompt, api_key, openai_model)
                    
                    # Display the analysis report in a nice format
                    st.markdown("### 📝 Analysis Report")
                    st.markdown(llm_response)
                    
                    # Add download button for the report
                    st.download_button(
                        label="📥 Download Report", 
                        data=f"# Statistical Analysis Report\n\n## Analysis Details\n- Analysis Type: {analysis_type}\n- Method: {method}\n\n## Results\n{st.session_state.analysis_results}\n\n## Expert Interpretation\n{llm_response}",
                        file_name="statistical_analysis_report.md",
                        mime="text/markdown"
                    )
    
    # Chat with Data Tab
    with tab2:
        st.title("💬 Chat with Your Data")
        
        # Check if API key is provided
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar settings to use the chat feature.")
        else:
            st.write("Ask questions about your data in natural language. The AI will analyze your data and provide insights.")
            
            # Example questions
            with st.expander("💡 Example Questions", expanded=True):
                example_questions = [
                    "What are the key statistics for this dataset?",
                    "What columns have the most missing values?",
                    "Show me the correlation between the numeric columns",
                    "What are the main insights from this data?",
                    "Are there any outliers in the dataset?",
                    "What's the distribution of [column_name]?",
                    "Compare the values of [column1] and [column2]"
                ]
                
                # Display example questions as buttons
                cols = st.columns(2)
                for i, question in enumerate(example_questions):
                    with cols[i % 2]:
                        st.button(
                            question, 
                            key=f"chat_question_{i}",
                            on_click=set_question,
                            args=(question,),
                            use_container_width=True
                        )
            
            # Text input for user question
            user_question = st.text_input(
                "Ask a question about your data:",
                key="chat_text_input",
                value=st.session_state.user_question
            )
            
            # Process user question
            if st.button("Ask") and user_question:
                with st.spinner("Analyzing your data..."):
                    # Create context information about the dataframe
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    df_info = buffer.getvalue()
                    
                    # Sample data
                    df_sample = df.head(5).to_string()
                    
                    # Statistical summary
                    try:
                        df_stats = df.describe(include='all').to_string()
                    except:
                        df_stats = "Unable to generate statistics for all columns."
                    
                    # Create the prompt
                    prompt = f"""
                    You are a data analysis assistant helping a user understand their dataset.
                    
                    Here is information about the dataframe:
                    
                    ```
                    # DataFrame Info:
                    {df_info}
                    
                    # DataFrame Sample (first 5 rows):
                    {df_sample}
                    
                    # DataFrame Statistics:
                    {df_stats}
                    ```
                    
                    The user has asked the following question about this data:
                    "{user_question}"
                    
                    Please provide a clear, detailed answer based on the data provided. 
                    Include relevant statistics or patterns you observe.
                    If you need to make assumptions due to limited information, state them clearly.
                    Format your response with markdown when helpful for readability.
                    """
                    
                    # Get response from LLM
                    response = query_llm(prompt, api_key, openai_model)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": user_question, "answer": response})
                    
                    # Clear the input field
                    st.session_state.user_question = ""
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("Conversation History")
                
                for i, chat in enumerate(st.session_state.chat_history):
                    # User question
                    st.markdown(f"**🙋 You asked:** {chat['question']}")
                    
                    # AI response
                    st.markdown(f"**🤖 Analysis:**")
                    st.markdown(chat['answer'])
                    
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
                
                # Option to clear chat history
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.experimental_rerun()
else:
    # No file uploaded yet
    with tab1:
        st.title("📊 Comprehensive Statistical Analyzer")
        st.info("Please upload a CSV or Excel file in the sidebar to begin the analysis.")
    
    with tab2:
        st.title("💬 Chat with Your Data")
        st.info("Please upload a CSV or Excel file in the sidebar to start chatting with your data.")