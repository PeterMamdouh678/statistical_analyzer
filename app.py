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

st.set_page_config(page_title="Comprehensive Statistical Analyzer", layout="wide")
st.title("📊 Comprehensive Statistical Analyzer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    analysis_type = st.sidebar.selectbox("Choose Analysis Type", [
        "Descriptive Statistics",
        "Inferential Statistics",
        "Non-parametric Tests",
        "Time Series Analysis",
        "Advanced / Multivariate Analysis"
    ])

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    if analysis_type == "Descriptive Statistics":
        method = st.sidebar.selectbox("Choose Method", ["Summary Stats", "Frequency Table", "Cross-tabulation", "Boxplot", "Histogram"])

        if method == "Summary Stats":
            st.subheader("Descriptive Summary")
            st.dataframe(df.describe(include='all'))

        elif method == "Frequency Table":
            col = st.sidebar.selectbox("Select column", all_cols)
            st.subheader(f"Frequency Table for {col}")
            st.dataframe(df[col].value_counts().reset_index().rename(columns={'index': col, col: 'Frequency'}))

        elif method == "Cross-tabulation":
            col1 = st.sidebar.selectbox("Column 1", all_cols)
            col2 = st.sidebar.selectbox("Column 2", all_cols)
            st.subheader(f"Cross-tab: {col1} vs {col2}")
            st.dataframe(pd.crosstab(df[col1], df[col2]))

        elif method == "Boxplot":
            col = st.sidebar.selectbox("Numeric column", numeric_cols)
            st.subheader(f"Boxplot of {col}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

        elif method == "Histogram":
            col = st.sidebar.selectbox("Numeric column", numeric_cols)
            st.subheader(f"Histogram of {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

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
            st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

        elif method == "T-test (2 sample)":
            col = st.sidebar.selectbox("Select numeric column", numeric_cols)
            group_col = st.sidebar.selectbox("Group by column (2 groups)", all_cols)
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                group1 = df[df[group_col] == groups[0]][col].dropna()
                group2 = df[df[group_col] == groups[1]][col].dropna()
                t_stat, p_val = stats.ttest_ind(group1, group2)
                st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
            else:
                st.warning("Column must have exactly 2 unique values")

        elif method == "Paired T-test":
            col1 = st.sidebar.selectbox("First sample", numeric_cols)
            col2 = st.sidebar.selectbox("Second sample", numeric_cols)
            t_stat, p_val = stats.ttest_rel(df[col1].dropna(), df[col2].dropna())
            st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

        elif method == "ANOVA":
            col = st.sidebar.selectbox("Numeric column", numeric_cols)
            group_col = st.sidebar.selectbox("Group by column", all_cols)
            groups = [group[col].dropna() for name, group in df.groupby(group_col)]
            f_stat, p_val = stats.f_oneway(*groups)
            st.write(f"F-statistic: {f_stat:.4f}, P-value: {p_val:.4f}")

        elif method == "Chi-Square Test":
            col1 = st.sidebar.selectbox("Column 1", all_cols)
            col2 = st.sidebar.selectbox("Column 2", all_cols)
            table = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = stats.chi2_contingency(table)
            st.write(f"Chi2: {chi2:.4f}, P-value: {p:.4f}, DOF: {dof}")

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

        elif method == "Confidence Interval":
            col = st.sidebar.selectbox("Select numeric column", numeric_cols)
            ci_level = st.sidebar.slider("Confidence level", 80, 99, 95)
            data = df[col].dropna()
            mean = data.mean()
            sem = stats.sem(data)
            margin = sem * stats.t.ppf((1 + ci_level / 100) / 2., len(data)-1)
            st.write(f"{ci_level}% CI for {col}: ({mean - margin:.4f}, {mean + margin:.4f})")

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
                st.write(f"U-statistic: {u_stat:.4f}, P-value: {p_val:.4f}")

        elif method == "Wilcoxon Signed-Rank Test":
            col1 = st.sidebar.selectbox("First sample", numeric_cols)
            col2 = st.sidebar.selectbox("Second sample", numeric_cols)
            w_stat, p_val = stats.wilcoxon(df[col1].dropna(), df[col2].dropna())
            st.write(f"W-statistic: {w_stat:.4f}, P-value: {p_val:.4f}")

        elif method == "Kruskal-Wallis H Test":
            col = st.sidebar.selectbox("Numeric column", numeric_cols)
            group_col = st.sidebar.selectbox("Group column", all_cols)
            groups = [group[col].dropna() for name, group in df.groupby(group_col)]
            h_stat, p_val = stats.kruskal(*groups)
            st.write(f"H-statistic: {h_stat:.4f}, P-value: {p_val:.4f}")

    elif analysis_type == "Time Series Analysis":
        col = st.sidebar.selectbox("Select time series column", numeric_cols)
        if col:
            st.line_chart(df[col])
            result = adfuller(df[col].dropna())
            st.write(f"ADF Statistic: {result[0]:.4f}, P-value: {result[1]:.4f}")
            st.write("Time Series is likely stationary" if result[1] < 0.05 else "Non-stationary")

    elif analysis_type == "Advanced / Multivariate Analysis":
        method = st.sidebar.selectbox("Choose Method", ["Linear Regression", "Logistic Regression", "PCA", "K-Means Clustering", "Random Forest"])

        if method == "Linear Regression":
            target = st.sidebar.selectbox("Target variable", numeric_cols)
            features = st.sidebar.multiselect("Feature variables", [col for col in numeric_cols if col != target])
            if features:
                X = df[features].dropna()
                y = df[target].loc[X.index]
                model = LinearRegression().fit(X, y)
                st.write("Coefficients:", dict(zip(features, model.coef_)))
                st.write(f"Intercept: {model.intercept_:.4f}, R-squared: {model.score(X, y):.4f}")

        elif method == "Logistic Regression":
            target = st.sidebar.selectbox("Target variable (binary)", all_cols)
            features = st.sidebar.multiselect("Feature variables", numeric_cols)
            if features:
                X = df[features].dropna()
                y = df[target].loc[X.index]
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                st.write("Coefficients:", dict(zip(features, model.coef_[0])))
                st.write(f"Intercept: {model.intercept_[0]:.4f}")

        elif method == "PCA":
            n_components = st.sidebar.slider("Number of components", 1, len(numeric_cols), 2)
            X = df[numeric_cols].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X_scaled)
            st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
            st.dataframe(pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)]))

        elif method == "K-Means Clustering":
            X = df[numeric_cols].dropna()
            k = st.sidebar.slider("Number of clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=k)
            X_scaled = StandardScaler().fit_transform(X)
            labels = kmeans.fit_predict(X_scaled)
            df['Cluster'] = labels
            st.write("Cluster Centers:", kmeans.cluster_centers_)
            st.dataframe(df.head())

        elif method == "Random Forest":
            target = st.sidebar.selectbox("Target variable", all_cols)
            features = st.sidebar.multiselect("Feature variables", numeric_cols)
            if features:
                X = df[features].dropna()
                y = df[target].loc[X.index]
                model = RandomForestClassifier()
                model.fit(X, y)
                importances = model.feature_importances_
                st.write("Feature Importances:", dict(zip(features, importances)))
