import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("This is a test sentence!"))

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
preprocessed_file_path_csv = os.path.join(downloads_folder, "latest_posts_cleaned.csv")
sentiment_output_file_csv = os.path.join(downloads_folder, "sentiment_analysis_results.csv")
sentiment_output_file_xlsx = os.path.join(downloads_folder, "sentiment_analysis_results.xlsx")
sentiment_chart_file_path = os.path.join(downloads_folder, "sentiment_distribution.png")

# Load the preprocessed data
print("Loading preprocessed data...")
data = pd.read_csv(preprocessed_file_path_csv)

# Initialize VADER SentimentIntensityAnalyzer
print("Initializing sentiment analyzer...")
sia = SentimentIntensityAnalyzer()

# Ensure that all entries in 'Processed_Text' are strings
data['Cleaned_Body'] = data['Cleaned_Body'].fillna("").astype(str)

# Apply sentiment analysis
print("Performing sentiment analysis...")
tqdm.pandas(desc="Analyzing sentiment")
data['Sentiment_Scores'] = data['Cleaned_Body '].progress_apply(lambda text: sia.polarity_scores(text))

# Extract individual sentiment components
data['Positive'] = data['Sentiment_Scores'].apply(lambda x: x['pos'])
data['Negative'] = data['Sentiment_Scores'].apply(lambda x: x['neg'])
data['Neutral'] = data['Sentiment_Scores'].apply(lambda x: x['neu'])
data['Compound'] = data['Sentiment_Scores'].apply(lambda x: x['compound'])

# Classify overall sentiment based on compound score
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

data['Sentiment'] = data['Compound'].apply(classify_sentiment)

# Save sentiment analysis results to CSV and Excel
print(f"Saving sentiment analysis results to {sentiment_output_file_csv} and {sentiment_output_file_xlsx}...")
data.to_csv(sentiment_output_file_csv, index=False)
data.to_excel(sentiment_output_file_xlsx, index=False)

# Plot sentiment distribution
print("Creating sentiment distribution chart...")
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Sentiment', palette='viridis', order=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig(sentiment_chart_file_path)
plt.show()

print(f"Sentiment distribution chart saved to {sentiment_chart_file_path}.")
print("Sentiment analysis complete!")
