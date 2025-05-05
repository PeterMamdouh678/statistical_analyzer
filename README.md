# 📊 Comprehensive Statistical Analyzer

A powerful and interactive Streamlit-based web application for performing a wide variety of statistical analyses—ranging from basic descriptive statistics to advanced multivariate modeling and time series analysis.

## 🔧 Features

This app enables users to upload their own CSV datasets and perform:

### 📈 Descriptive Statistics

* **Summary Statistics** (mean, median, std. dev, etc.)
* **Frequency Tables** for categorical variables
* **Cross-tabulation** between two categorical variables
* **Boxplots** for visualizing distribution and outliers
* **Histograms** with KDE overlays

### 📈 Inferential Statistics

* **T-tests** (1-sample, 2-sample, and paired)
* **ANOVA** for comparing means across multiple groups
* **Chi-square test** for categorical variables
* **Correlation Matrix** (Pearson or Spearman)
* **Confidence Intervals** for numeric variables

### 🧪 Non-parametric Tests

* **Mann-Whitney U Test** (non-parametric 2-sample test)
* **Wilcoxon Signed-Rank Test** (paired test)
* **Kruskal-Wallis H Test** (non-parametric ANOVA)

### ⏱️ Time Series Analysis

* **ADF Test** for stationarity
* **Time Series Line Plots**
* **ACF/PACF plots** (can be extended)

### 🧬 Advanced / Multivariate Analysis

* **Linear Regression**
* **Logistic Regression**
* **Principal Component Analysis (PCA)**
* **K-Means Clustering**
* **Random Forest Classification** with feature importance

---

## 🚀 How to Run the App

1. **Install required libraries:**

   ```bash
   pip install streamlit pandas numpy seaborn matplotlib scipy scikit-learn statsmodels
   ```

2. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

3. **Upload a CSV file** using the file uploader.

---

## 📂 Folder Structure

```
.
├── app.py                  # Main Streamlit app
└── README.md               # This documentation file
```

---

## 📌 Notes

* Ensure your CSV file has a proper header row.
* For **Logistic Regression**, the target column should be binary.
* For **K-means** and **PCA**, only numeric columns are used.
* The app handles missing values by dropping rows in analysis-specific steps.

---

## 📍 To Do / Future Improvements

* Add **ACF/PACF plots** in the Time Series section
* Enable **model saving/exporting**
* Include **missing value imputation** options
* Add **interactive plots** using Plotly

---

## 🧑‍💻 Author

Built with ❤️ using [Streamlit](https://streamlit.io/) by Peter Mamdouh.
