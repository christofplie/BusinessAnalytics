import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
cleaned_input_csv = os.path.join(downloads_folder, "latest_posts_cleaned.csv")

# Define output paths
outputs = {
    "unigram": {
        "xlsx": os.path.join(downloads_folder, "unigram_summary_old.xlsx"),
        "png": os.path.join(downloads_folder, "unigram_frequency_plot_old.png"),
    },
    "bigram": {
        "xlsx": os.path.join(downloads_folder, "bigram_summary_old.xlsx"),
        "png": os.path.join(downloads_folder, "bigram_frequency_plot_old.png"),
    },
    "trigram": {
        "xlsx": os.path.join(downloads_folder, "trigram_summary_old.xlsx"),
        "png": os.path.join(downloads_folder, "trigram_frequency_plot_old.png"),
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

# TF-IDF Analysis
print("Creating TF-IDF matrix...")
tfidf_vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(english_data['Processed_Text'])
terms = tfidf_vectorizer.get_feature_names_out()

# 1. Key Term Analysis
print("Performing key term analysis...")
key_terms_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
key_terms_df['Document_ID'] = english_data.index
key_terms_df = key_terms_df.melt(id_vars=['Document_ID'], var_name='Term', value_name='TF-IDF')
key_terms_df = key_terms_df[key_terms_df['TF-IDF'] > 0].sort_values(by='TF-IDF', ascending=False)
key_terms_path = os.path.join(downloads_folder, "key_term_analysis.csv")
key_terms_df.to_csv(key_terms_path, index=False)
print(f"Key term analysis saved to {key_terms_path}.")

# 2. Document Similarity Analysis
print("Performing document similarity analysis...")
cosine_sim_matrix = cosine_similarity(tfidf_matrix)
similarity_path = os.path.join(downloads_folder, "document_similarity.csv")
similarity_df = pd.DataFrame(cosine_sim_matrix, index=english_data.index, columns=english_data.index)
similarity_df.to_csv(similarity_path)
print(f"Document similarity matrix saved to {similarity_path}.")

# 3. Clustering/Topic Modeling
print("Performing clustering analysis...")
num_clusters = 10  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
english_data['Cluster'] = clusters
clustered_path = os.path.join(downloads_folder, "clustered_documents.csv")
english_data.to_csv(clustered_path, index=False)
print(f"Clustered documents saved to {clustered_path}.")

# 4. Dimensionality Reduction for Visualization
print("Performing dimensionality reduction...")
pca = PCA(n_components=2, random_state=42)
reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
reduced_df = pd.DataFrame(reduced_matrix, columns=['PC1', 'PC2'])
reduced_df['Cluster'] = clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100)
plt.title("Document Clusters (PCA Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
clusters_vis_path = os.path.join(downloads_folder, "clusters_visualization.png")
plt.savefig(clusters_vis_path)
plt.close()
print(f"Cluster visualization saved to {clusters_vis_path}.")

# Define new Excel file paths
analysis_results_excel = os.path.join(downloads_folder, "analysis_results.xlsx")

# Initialize Excel writer
with pd.ExcelWriter(analysis_results_excel, engine='openpyxl') as writer:
    # 1. Key Term Analysis
    print("Saving key term analysis to Excel...")
    key_terms_df.to_excel(writer, sheet_name="Key_Term_Analysis", index=False)

    # 2. Document Similarity Analysis
    print("Saving document similarity analysis to Excel...")
    similarity_df.to_excel(writer, sheet_name="Document_Similarity")

    # 3. Clustering/Topic Modeling
    print("Saving clustering results to Excel...")
    cluster_summary = english_data[['Processed_Text', 'Cluster']]
    cluster_summary.to_excel(writer, sheet_name="Clusters", index=False)

    # 4. Dimensionality Reduction for Visualization
    print("Saving PCA results to Excel...")
    reduced_df.to_excel(writer, sheet_name="PCA_Reduction", index=False)

print(f"All analysis results saved to {analysis_results_excel}.")

print("All analyses completed.")
