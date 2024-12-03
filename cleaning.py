import os
import re
import pandas as pd
import spacy
spacy.cli.download("en_core_web_sm")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
input_file_path = os.path.join(downloads_folder, "latest_posts.csv")
preprocessed_file_path_csv = os.path.join(downloads_folder, "latest_posts_cleaned_spacy.csv")
preprocessed_file_path_xlsx = os.path.join(downloads_folder, "latest_posts_cleaned_spacy.xlsx")

# 1. Load the dataset
data = pd.read_csv(input_file_path)
data = data.dropna(subset=["Body"]).reset_index(drop=True)

# 2. Clean the text
def clean_text(text):
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\W+', ' ', text)    # Remove special characters
    text = text.lower()                 # Convert to lowercase
    return text

data['Cleaned_Body'] = data['Body'].apply(clean_text)

# 3. Tokenization and lemmatization with SpaCy
def process_text_spacy(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]  # Lemmatize, remove stopwords
    return tokens

data['Lemmatized_Tokens'] = data['Cleaned_Body'].apply(process_text_spacy)

# 4. Generate TF-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer

data['Processed_Text'] = data['Lemmatized_Tokens'].apply(lambda tokens: ' '.join(tokens))  # Join tokens into text
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(data['Processed_Text'])

# Convert TF-IDF matrix to DataFrame for inspection
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Save the preprocessed data
data.to_csv(preprocessed_file_path_csv, index=False)
data.to_excel(preprocessed_file_path_xlsx, index=False)
print(f"Preprocessed data saved to {preprocessed_file_path_csv}")
