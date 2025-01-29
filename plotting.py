import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
result_file_path = os.path.join(downloads_folder, "sentiment_analysis_results.xlsx")
gpt_sentiment_file_path = os.path.join(downloads_folder, "gpt_sentiment.csv")

# Load the datasets
result_data = pd.read_excel(result_file_path)
gpt_data = pd.read_csv(gpt_sentiment_file_path)

# Ensure 'GPT_Sentiment' and 'GPT_Sentiment_Category' are in gpt_sentiment file
if not {"GPT_Sentiment", "GPT_Sentiment_Category"}.issubset(gpt_data.columns):
    raise ValueError("Missing columns 'GPT_Sentiment' or 'GPT_Sentiment_Category' in GPT sentiment file.")

# Merge data based on a common column
merged_data = pd.concat([result_data.reset_index(drop=True), gpt_data[["GPT_Sentiment", "GPT_Sentiment_Category"]].reset_index(drop=True)], axis=1)

# Save the merged result (optional)
merged_output_path = os.path.join(downloads_folder, "merged_sentiment_analysis.xlsx")
merged_data.to_excel(merged_output_path, index=False)
print(f"Merged file saved to: {merged_output_path}")

# Standardize GPT sentiment labels to match the format of the Sentiment column
merged_data['GPT_Sentiment_Category'] = merged_data['GPT_Sentiment_Category'].str.capitalize()

# Plot occurrences in 'Sentiment' and 'GPT_Sentiment_Category' columns
for sentiment_column, title, filename in [
    ("Sentiment", "Occurrences of Sentiment Categories (Sentiment Analysis)", "sentiment_occurrences.png"),
    ("GPT_Sentiment_Category", "Occurrences of Sentiment Categories (GPT Sentiment)", "gpt_sentiment_occurrences.png")
]:
    if sentiment_column not in merged_data.columns:
        continue

    plt.figure(figsize=(8, 6))

    # Custom color palette for sentiments
    sentiment_colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}

    # Use Seaborn countplot for better aesthetics
    sns.countplot(data=merged_data, x=sentiment_column, palette=sentiment_colors,
                  order=['Positive', 'Neutral', 'Negative'])

    plt.title(title)
    plt.xlabel("Sentiment Category")
    plt.ylabel("Count")
    plt.ylim(0, 2000)
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(downloads_folder, filename))
    plt.show()

# Create confusion matrix
conf_matrix = confusion_matrix(merged_data['Sentiment'], merged_data['GPT_Sentiment_Category'], labels=['Positive', 'Neutral', 'Negative'])

# Display confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title('Confusion Matrix Sentiment')
plt.xlabel('Predicted Sentiment (GPT Sentiment)')
plt.ylabel('True Sentiment (Original Sentiment)')
plt.tight_layout()

# Save and show the confusion matrix plot
confusion_matrix_file_path = os.path.join(downloads_folder, "confusion_matrix.png")
plt.savefig(confusion_matrix_file_path)
plt.show()
print(f"Confusion matrix saved to: {confusion_matrix_file_path}")

