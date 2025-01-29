import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# File paths
downloads_folder = os.path.expanduser("~/Downloads")
cleaned_input_csv = os.path.join(downloads_folder, "latest_posts_cleaned.csv")

# Load data
print("Loading preprocessed data...")
english_data = pd.read_csv(cleaned_input_csv)
cleaned_data = english_data.dropna(subset=['Body']).reset_index(drop=True)
texts = cleaned_data['Body']

# Load BERT model
print("Loading BERT model...")
model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# Calculate Cosine Similarity
print("Calculating document similarity...")
cosine_sim_matrix = cosine_similarity(embeddings)
similarity_df = pd.DataFrame(cosine_sim_matrix, index=cleaned_data.index, columns=cleaned_data.index)
similarity_path = os.path.join(downloads_folder, "bert_document_similarity.csv")
similarity_df.to_csv(similarity_path)
print(f"Document similarity matrix saved to {similarity_path}.")

# Clustering
print("Performing clustering...")
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)
cleaned_data['Cluster'] = clusters


# Cluster analysis functions
def get_cluster_keywords(cluster_texts, n_keywords=10):
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = tfidf.fit_transform(cluster_texts)
    feature_names = tfidf.get_feature_names_out()

    # Sum TF-IDF scores across documents and get top indices
    summed_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(summed_scores)[::-1]

    # Return actual words sorted by importance
    return [feature_names[i] for i in sorted_indices[:n_keywords]]


def get_representative_docs(embeddings, clusters, n_examples=3):
    cluster_centers = []
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_center = np.mean(embeddings[mask], axis=0)
        cluster_centers.append(cluster_center)

    cluster_centers = np.array(cluster_centers)
    similarities = cosine_similarity(embeddings, cluster_centers)

    representative_docs = defaultdict(list)
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_sims = similarities[cluster_indices, cluster_id]
        top_indices = cluster_indices[np.argsort(cluster_sims)[-n_examples:]]
        representative_docs[cluster_id] = top_indices.tolist()

    return representative_docs

# Perform cluster analysis
print("\nAnalyzing cluster topics...")
representative_docs = get_representative_docs(embeddings, clusters)
cluster_analysis = []
cluster_keywords = {}

for cluster_id in sorted(cleaned_data['Cluster'].unique()):
    cluster_texts = cleaned_data[cleaned_data['Cluster'] == cluster_id]['Body'].tolist()

    # Get keywords
    keywords = get_cluster_keywords(cluster_texts)
    cluster_keywords[cluster_id] = keywords

    # Get examples
    example_indices = representative_docs.get(cluster_id, [])
    examples = [cleaned_data.iloc[idx]['Body'] for idx in example_indices] if example_indices else []

    # Store analysis
    cluster_analysis.append({
        'Cluster': cluster_id,
        'Size': len(cluster_texts),
        'Top_Keywords': ", ".join(keywords),
        'Example_1': examples[0] if len(examples) > 0 else "",
        'Example_2': examples[1] if len(examples) > 1 else "",
        'Example_3': examples[2] if len(examples) > 2 else ""
    })

# Save cluster analysis
cluster_analysis_df = pd.DataFrame(cluster_analysis)
analysis_path = os.path.join(downloads_folder, "cluster_analysis.csv")
cluster_analysis_df.to_csv(analysis_path, index=False)
print(f"Cluster analysis saved to {analysis_path}")

# Visualization with proper keyword labels
print("Creating visualizations...")
plt.figure(figsize=(14, 10))
for cluster_id, keywords in cluster_keywords.items():
    plt.subplot(2, 2, cluster_id + 1)

    # Create horizontal bar plot with actual words
    sns.barplot(x=np.arange(len(keywords)), y=keywords, palette='viridis', orient='h')

    plt.title(f'Cluster {cluster_id} - Top Keywords', fontsize=12, pad=15)
    plt.xlabel('TF-IDF Score Rank', fontsize=10)
    plt.ylabel('Keywords', fontsize=10)
    plt.yticks(fontsize=9)
    plt.xticks([])  # Hide numeric x-ticks since we're showing rank

plt.tight_layout(pad=3.0)
keywords_vis_path = os.path.join(downloads_folder, "cluster_keywords.png")
plt.savefig(keywords_vis_path, bbox_inches='tight')
plt.close()

# Dimensionality reduction for visualization
print("Performing dimensionality reduction...")
pca = PCA(n_components=2, random_state=42)
reduced_matrix = pca.fit_transform(embeddings)
reduced_df = pd.DataFrame(reduced_matrix, columns=['PC1', 'PC2'])
reduced_df['Cluster'] = clusters

# Scatterplot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100)
plt.title("Document Clusters (BERT Embeddings Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
clusters_vis_path = os.path.join(downloads_folder, "bert_clusters_visualization.png")
plt.savefig(clusters_vis_path)
plt.close()

# Save all results to Excel
analysis_results_excel = os.path.join(downloads_folder, "bert_analysis_results.xlsx")
with pd.ExcelWriter(analysis_results_excel, engine='openpyxl') as writer:
    print("Saving all results to Excel...")
    similarity_df.to_excel(writer, sheet_name="Document_Similarity", index=False)
    cleaned_data[['Body', 'Cluster']].to_excel(writer, sheet_name="Clusters", index=False)
    reduced_df.to_excel(writer, sheet_name="PCA_Reduction", index=False)
    cluster_analysis_df.to_excel(writer, sheet_name="Cluster_Analysis", index=False)

print(f"All analysis results saved to {analysis_results_excel}")
print("BERT-based analysis completed.")
