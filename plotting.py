import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
downloads_folder = os.path.expanduser("~/Downloads")
file_path = os.path.join(downloads_folder, "sentiment_analysis_results.xlsx")
data = pd.read_excel(file_path)

# Ensure the relevant columns are present
if 'Sentiment' not in data.columns or 'GPT_Sentiment_Category' not in data.columns:
    raise ValueError("The required columns ('sentiment', 'GPT_Sentiment_Category') are not present in the file.")

# Define the desired order of categories
category_order = ['Positive', 'Neutral', 'Negative']

# Count occurrences for each category
sentiment_counts = data['Sentiment'].value_counts()
gpt_sentiment_counts = data['GPT_Sentiment_Category'].value_counts()


# Plot 1: Occurrences in 'sentiment' column
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['lightgreen', 'orange', 'skyblue'])
plt.title("Occurrences of Sentiment Categories (Sentiment Analysis)")
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.ylim(0, 2000)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_occurrences.png")  # Save the plot (optional)
plt.show()

# Plot 2: Occurrences in 'GPT_Sentiment_Category' column
plt.figure(figsize=(8, 6))
gpt_sentiment_counts.plot(kind='bar', color=['skyblue', 'orange', 'lightgreen'])
plt.title("Occurrences of Sentiment Categories (GPT Sentiment)")
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.ylim(0, 2000)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("gpt_sentiment_occurrences.png")  # Save the plot (optional)
plt.show()
