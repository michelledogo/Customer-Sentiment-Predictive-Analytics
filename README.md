# Customer-Sentiment-Predictive-Analytics
A machine learning and NLP project that analyzes customer reviews and predicts sentiment as positive, neutral, or negative to support business decision-making

## Project Overview

Businesses receive thousands of customer comments, reviews, and social media posts every day. Reading each review manually is slow, inconsistent, and difficult to scale. This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify customer reviews as **positive**, **neutral**, or **negative**.

The goal is to help businesses understand customer opinions, identify dissatisfaction early, and make better decisions based on real customer feedback.

## Problem Statement

Customer feedback contains valuable information about satisfaction, product quality, service problems, and brand perception. However, manually reviewing feedback can be time-consuming and biased. This project builds a predictive sentiment analysis model that can classify reviews and provide visual insights for business decision-making.

## Project Objectives

- Collect and clean real-world customer review data.
- Perform exploratory data analysis to understand review patterns.
- Convert text data into numerical features using TF-IDF.
- Train multiple machine learning models for sentiment classification.
- Compare models using accuracy, precision, recall, and F1-score.
- Deploy the best model using a simple Streamlit web application.
- Explain how the results can support business decisions.

## Dataset

This project is designed to work with public sentiment datasets such as:

- Twitter US Airline Sentiment Dataset from Kaggle
- Yelp customer reviews
- Amazon product reviews
- Trustpilot or Google review exports

For demonstration, the project includes a small sample dataset in `data/raw/sample_reviews.csv`.

Recommended Kaggle dataset:
CrowdFlower. (2015). Twitter US Airline Sentiment. Kaggle.

## Project Structure

```text
customer_sentiment_predictive_analytics/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── sample_reviews.csv
│   └── processed/
│
├── models/
│
├── notebooks/
│   └── sentiment_analysis_project.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   └── predict.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── main.py
```

## Methodology

### 1. Data Collection

Customer reviews can be collected from Kaggle datasets, customer surveys, online reviews, or social media data. The dataset should include at least two columns:

- `review`: the text of the customer feedback
- `sentiment`: the target label, such as positive, neutral, or negative

### 2. Data Preprocessing

The text is cleaned before modeling. The preprocessing steps include:

- Converting text to lowercase
- Removing links, punctuation, numbers, and special characters
- Removing extra spaces
- Removing stopwords
- Lemmatizing words

### 3. Exploratory Data Analysis

EDA helps understand the dataset before modeling. The analysis includes:

- Sentiment distribution
- Review length distribution
- Most frequent words
- Word clouds by sentiment class
- Class imbalance analysis

### 4. Feature Engineering

The project uses **TF-IDF vectorization** to convert customer reviews into numerical features. TF-IDF is useful because it gives higher importance to words that are meaningful within a review but not too common across all reviews.

### 5. Model Training

The following models are trained and compared:

- Logistic Regression
- Naive Bayes
- Support Vector Machine
- Random Forest

### 6. Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The final model is selected based mainly on F1-score because sentiment datasets can be imbalanced.

### 7. Deployment

The best model is saved and used in a Streamlit application. The app allows a user to enter a customer review and instantly receive a predicted sentiment.

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/customer_sentiment_predictive_analytics.git
cd customer_sentiment_predictive_analytics
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

For Mac/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python main.py
```

### Step 5: Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Expected Results

The final output includes:

- A trained sentiment classification model
- A model performance comparison table
- A confusion matrix
- A saved machine learning pipeline
- A Streamlit web app for live predictions

## Business Value

This project can help businesses:

- Detect negative customer feedback earlier
- Prioritize urgent customer service issues
- Track customer satisfaction trends
- Improve products and services
- Support marketing and brand management decisions
- Reduce manual review time

## Limitations

- The model depends on the quality and size of the training data.
- Sarcasm and slang may be difficult to classify correctly.
- A model trained on one industry may not always perform well in another.
- Customer reviews may contain bias or incomplete information.

## Future Improvements

- Use BERT or other transformer models for better contextual understanding.
- Add multilingual sentiment analysis.
- Create a dashboard with time-based sentiment trends.
- Connect the model to live review platforms or business intelligence tools.
- Add topic modeling to identify the main reasons behind customer dissatisfaction.

## References

CrowdFlower. (2015). *Twitter US airline sentiment* [Data set]. Kaggle.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171–4186.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.

Medhat, W., Hassan, A., & Korashy, H. (2014). Sentiment analysis algorithms and applications: A survey. *Ain Shams Engineering Journal, 5*(4), 1093–1113.

Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval, 2*(1–2), 1–135.
