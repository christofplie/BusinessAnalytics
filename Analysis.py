import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# File paths
downloads_folder = os.path.expanduser("~/Downloads")
cleaned_input_csv = os.path.join(downloads_folder, "latest_posts_cleaned.csv")

# Load data
print("Loading preprocessed data...")
english_data = pd.read_csv(cleaned_input_csv)

# Use the Cleaned_Body column for analysis
cleaned_data = english_data.dropna(subset=['Body']).reset_index(drop=True)

# Extract texts for embedding
texts = cleaned_data['Body']

# Load a pre-trained BERT model (Sentence-BERT recommended for semantic similarity)
print("Loading BERT model...")
model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# 1. Calculate Cosine Similarity
print("Calculating document similarity...")
cosine_sim_matrix = cosine_similarity(embeddings)
similarity_df = pd.DataFrame(cosine_sim_matrix, index=cleaned_data.index, columns=cleaned_data.index)
similarity_path = os.path.join(downloads_folder, "bert_document_similarity.csv")
similarity_df.to_csv(similarity_path)
print(f"Document similarity matrix saved to {similarity_path}.")

# 2. Clustering/Topic Modeling
from sklearn.cluster import KMeans

print("Performing clustering...")
num_clusters = 4  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)
cleaned_data['Cluster'] = clusters
clustered_path = os.path.join(downloads_folder, "bert_clustered_documents.csv")
cleaned_data.to_csv(clustered_path, index=False)
print(f"Clustered documents saved to {clustered_path}.")

# 3. Dimensionality Reduction for Visualization
print("Performing dimensionality reduction...")
from sklearn.decomposition import PCA

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
print(f"Cluster visualization saved to {clusters_vis_path}.")

# Save results to an Excel file
analysis_results_excel = os.path.join(downloads_folder, "bert_analysis_results.xlsx")
with pd.ExcelWriter(analysis_results_excel, engine='openpyxl') as writer:
    print("Saving key term and clustering results to Excel...")
    similarity_df.to_excel(writer, sheet_name="Document_Similarity", index=False)
    cluster_summary = cleaned_data[['Cleaned_Body', 'Cluster']]
    cluster_summary.to_excel(writer, sheet_name="Clusters", index=False)
    reduced_df.to_excel(writer, sheet_name="PCA_Reduction", index=False)
print(f"All analysis results saved to {analysis_results_excel}.")

print("BERT-based analysis completed.")