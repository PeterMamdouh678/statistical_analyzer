# Comprehensive Statistical Analyzer

A powerful statistical analysis and data exploration tool built with Streamlit and enhanced with AI capabilities.

## Features

- **Data Upload**: Support for CSV and Excel files (.xlsx, .xls)
- **Statistical Analysis**: Wide range of statistical methods and tests
- **AI-Powered Reports**: Generate detailed analysis reports with explanations using OpenAI's language models
- **Chat with Data**: Ask natural language questions about your data and get AI-generated insights
- **Interactive Visualizations**: Generated automatically based on your data and analysis type
- **Export Options**: Download reports and analysis results

## Analysis Types

### Descriptive Statistics
- Summary Statistics
- Frequency Tables
- Cross-tabulations
- Boxplots
- Histograms

### Inferential Statistics
- T-tests (1-sample, 2-sample, paired)
- ANOVA
- Chi-Square Tests
- Correlation Analysis
- Confidence Intervals

### Non-parametric Tests
- Mann-Whitney U Test
- Wilcoxon Signed-Rank Test
- Kruskal-Wallis H Test

### Time Series Analysis
- Stationarity Tests
- Time Series Visualization

### Advanced/Multivariate Analysis
- Linear Regression
- Logistic Regression
- Principal Component Analysis (PCA)
- K-Means Clustering
- Random Forest Feature Importance

## Requirements

```
streamlit
pandas
numpy
seaborn
matplotlib
scipy
scikit-learn
statsmodels
openai
```

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Statistical Analysis

1. Upload your data file (CSV or Excel) using the sidebar uploader
2. Select the type of analysis you want to perform
3. Choose the specific method or test
4. Configure any method-specific options (columns, parameters, etc.)
5. View the results directly in the app
6. Click "Generate AI Analysis Report" to get an AI-powered explanation of your results

### Chat with Data

1. Upload your data file (CSV or Excel) using the sidebar uploader
2. Navigate to the "Chat with Data" tab
3. Enter your OpenAI API key in the sidebar settings
4. Ask questions about your data in natural language
5. View the AI-generated responses with insights about your data

## AI Integration

This application uses OpenAI's API to provide AI-powered features:

1. **Analysis Reports**: Get detailed explanations of statistical test results with practical interpretations
2. **Data Chat**: Ask questions about your data in plain English and get insightful responses

To use these features, you need to provide your own OpenAI API key in the sidebar settings.

## Privacy & Security

- Your data is processed locally and not stored permanently
- API keys are only used for the current session and not saved
- No data is shared with third parties except when sending queries to OpenAI's API

## Limitations

- Large datasets may cause performance issues
- Complex visualizations may take time to render
- AI features require an internet connection and a valid OpenAI API key
- The quality of AI analysis depends on the OpenAI model selected

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- AI-powered features use [OpenAI API](https://openai.com/)
- Statistical analysis with [SciPy](https://scipy.org/), [Pandas](https://pandas.pydata.org/), and [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)