import os
import re
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
input_file_path = os.path.join(downloads_folder, "latest_posts.csv")

# Load the dataset
data = pd.read_csv(input_file_path)
data = data.dropna(subset=["Body"]).reset_index(drop=True)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()  # Average over token embeddings

# Generate embeddings
data['Cleaned_Body'] = data['Body'].apply(lambda text: re.sub(r'<[^>]*>', '', text))  # Basic cleaning
data['BERT_Embedding'] = data['Cleaned_Body'].apply(get_bert_embeddings)

# Convert embeddings to DataFrame
bert_embeddings_df = pd.DataFrame(data['BERT_Embedding'].tolist())

# Save embeddings for analysis
bert_embeddings_path = os.path.join(downloads_folder, "bert_embeddings.csv")
bert_embeddings_df.to_csv(bert_embeddings_path, index=False)
print(f"BERT embeddings saved to {bert_embeddings_path}")
