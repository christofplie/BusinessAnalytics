import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
cleaned_input_csv = os.path.join(downloads_folder, "latest_posts_cleaned.csv")

# Define output paths
outputs = {
    "unigram": {
        "xlsx": os.path.join(downloads_folder, "unigram_summary.xlsx"),
        "png": os.path.join(downloads_folder, "unigram_frequency_plot.png"),
    },
    "bigram": {
        "xlsx": os.path.join(downloads_folder, "bigram_summary.xlsx"),
        "png": os.path.join(downloads_folder, "bigram_frequency_plot.png"),
    },
    "trigram": {
        "xlsx": os.path.join(downloads_folder, "trigram_summary.xlsx"),
        "png": os.path.join(downloads_folder, "trigram_frequency_plot.png"),
    },
}

# Load preprocessed data
print("Loading preprocessed data...")
english_data = pd.read_csv(cleaned_input_csv)

# Prepare text for analysis
print("Preparing text for analysis...")
english_data['Processed_Text'] = english_data['Lemmatized_Tokens'].apply(lambda tokens: ' '.join(eval(tokens)))

# Function to perform n-gram analysis
def perform_ngram_analysis(texts, ngram_range, output_paths, title_prefix):
    print(f"Performing {title_prefix.lower()} analysis...")

    # Generate n-gram matrix
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=500)
    ngram_matrix = vectorizer.fit_transform(texts)

    # Create summary
    ngram_df = pd.DataFrame(ngram_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    ngram_summary = ngram_df.sum().sort_values(ascending=False).reset_index()
    ngram_summary.columns = ['N-gram', 'Frequency']

    # Save summary to Excel
    ngram_summary.to_excel(output_paths['xlsx'], index=False)
    print(f"{title_prefix} summary saved to {output_paths['xlsx']}.")

    # Visualize top 20 n-grams
    top_ngrams = ngram_summary.head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_ngrams, x='Frequency', y='N-gram', hue='N-gram', dodge=False, legend=False)
    plt.title(f'Top 20 Most Frequent {title_prefix}')
    plt.xlabel('Frequency')
    plt.ylabel(f'{title_prefix}')
    plt.tight_layout()
    plt.savefig(output_paths['png'])
    plt.close()  # Close the figure to prevent overlapping plots
    print(f"{title_prefix} frequency plot saved to {output_paths['png']}.")

# Perform analysis for unigrams, bigrams, and trigrams
perform_ngram_analysis(english_data['Processed_Text'], (1, 1), outputs["unigram"], "Unigrams")
perform_ngram_analysis(english_data['Processed_Text'], (2, 2), outputs["bigram"], "Bigrams")
perform_ngram_analysis(english_data['Processed_Text'], (3, 3), outputs["trigram"], "Trigrams")
