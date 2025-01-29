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

# Ensure required columns are available
if not {"GPT_Sentiment", "GPT_Sentiment_Category"}.issubset(gpt_data.columns):
    raise ValueError("Missing columns 'GPT_Sentiment' or 'GPT_Sentiment_Category' in GPT sentiment file.")

# Merge data based on a common column
merged_data = pd.concat([result_data.reset_index(drop=True), gpt_data[["GPT_Sentiment", "GPT_Sentiment_Category"]].reset_index(drop=True)], axis=1)

# Save the merged result (optional)
merged_output_path = os.path.join(downloads_folder, "merged_sentiment_analysis.xlsx")
merged_data.to_excel(merged_output_path, index=False)
print(f"Merged file saved to: {merged_output_path}")

# Standardize GPT sentiment labels
merged_data['GPT_Sentiment_Category'] = merged_data['GPT_Sentiment_Category'].str.capitalize()

# Plot occurrences in sentiment columns
sentiment_columns = [
    ("Sentiment", "Occurrences of Sentiment Categories (Original)", "sentiment_occurrences.png"),
    ("GPT_Sentiment_Category", "Occurrences of Sentiment Categories (GPT Sentiment)", "gpt_sentiment_occurrences.png"),
    ("Sentiment_Custom", "Occurrences of Sentiment Categories (Custom)", "custom_sentiment_occurrences.png")
]

for sentiment_column, title, filename in sentiment_columns:
    if sentiment_column not in merged_data.columns:
        continue

    plt.figure(figsize=(8, 6))

    sentiment_colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}

    sns.countplot(data=merged_data, x=sentiment_column, palette=sentiment_colors,
                  order=['Positive', 'Neutral', 'Negative'])

    # Formatting adjustments
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Sentiment Category", fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel("Count", fontsize=12, fontweight='bold', labelpad=15)

    plt.xticks(fontsize=11, rotation=0)
    plt.yticks(fontsize=11)

    plt.ylim(0, 2000)
    plt.tight_layout(pad=3.0)

    plt.subplots_adjust(left=0.15, bottom=0.15)

    plt.savefig(os.path.join(downloads_folder, filename), bbox_inches='tight')
    plt.show()

# Create confusion matrices and plots
comparison_pairs = [
    ("Sentiment", "GPT_Sentiment_Category", "Original vs GPT"),
    ("Sentiment", "Sentiment_Custom", "Original vs Custom"),
    ("GPT_Sentiment_Category", "Sentiment_Custom", "GPT vs Custom")
]

# Create label mapping for display names
label_mapping = {
    "Sentiment": "Original",
    "GPT_Sentiment_Category": "GPT",
    "Sentiment_Custom": "Custom"
}

for true_col, pred_col, title in comparison_pairs:
    if true_col not in merged_data.columns or pred_col not in merged_data.columns:
        continue

    conf_matrix = confusion_matrix(merged_data[true_col], merged_data[pred_col],
                                   labels=['Positive', 'Neutral', 'Negative'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'])

    # Get display names from mapping
    true_label = label_mapping.get(true_col, true_col)
    pred_label = label_mapping.get(pred_col, pred_col)

    plt.title(f'Confusion Matrix: {title}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(f'Predicted Sentiment ({pred_label})',
                fontweight='bold', labelpad=15)
    plt.ylabel(f'True Sentiment ({true_label})',
                fontweight='bold', labelpad=15)

    plt.tight_layout(pad=3.0)

    plt.subplots_adjust(left=0.2, bottom=0.2)

    confusion_matrix_file_path = os.path.join(downloads_folder,
                                              f"confusion_matrix_{title.replace(' ', '_').lower()}.png")
    plt.savefig(confusion_matrix_file_path, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix for {title} saved to: {confusion_matrix_file_path}")

