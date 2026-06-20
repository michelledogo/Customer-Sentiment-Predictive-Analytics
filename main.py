import pandas as pd
from src.data_preprocessing import download_nltk_resources, clean_text
from src.eda import plot_sentiment_distribution, plot_review_length, create_wordcloud
from src.model_training import train_and_compare_models

def main():
    download_nltk_resources()

    df = pd.read_csv("data/raw/sample_reviews.csv")

    print("Dataset Preview:")
    print(df.head())

    df["clean_review"] = df["review"].apply(clean_text)
    df.to_csv("data/processed/clean_reviews.csv", index=False)

    plot_sentiment_distribution(df)
    plot_review_length(df)
    create_wordcloud(df)

    best_name, best_model, results_df = train_and_compare_models(df)

    print("\nModel Comparison:")
    print(results_df)
    print(f"\nBest Model: {best_name}")
    print("\nTraining complete. Model saved in models/best_sentiment_model.joblib")

if __name__ == "__main__":
    main()