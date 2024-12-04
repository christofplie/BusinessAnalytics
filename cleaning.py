import os
import re
import pandas as pd
import spacy
import contractions
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Fix randomness in langdetect
DetectorFactory.seed = 0

# Load SpaCy model for English
nlp = spacy.load("en_core_web_sm")

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
input_file_path = os.path.join(downloads_folder, "latest_posts.csv")
preprocessed_file_path_csv = os.path.join(downloads_folder, "latest_posts_cleaned_spacy.csv")
preprocessed_file_path_xlsx = os.path.join(downloads_folder, "latest_posts_cleaned_spacy.xlsx")
non_english_file_path_xlsx = os.path.join(downloads_folder, "non_english_posts.xlsx")
tfidf_output_xlsx = os.path.join(downloads_folder, "tfidf_matrix.xlsx")

# 1. Load the dataset
print("Loading dataset...")
data = pd.read_csv(input_file_path)
data = data.dropna(subset=["Body"]).reset_index(drop=True)

# 2. Expand contractions
print("Expanding contractions...")
def expand_contractions(text):
    return contractions.fix(text)

data['Body_Expanded'] = data['Body'].apply(expand_contractions)

# 3. Detect and filter by language
print("Detecting language...")
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

# Add a language column
tqdm.pandas(desc="Detecting language")
data['Language'] = data['Body_Expanded'].apply(detect_language)

# Separate English and non-English rows
print("Separating non-English rows...")
non_english_data = data[data['Language'] != 'en'].reset_index(drop=True)
english_data = data[data['Language'] == 'en'].reset_index(drop=True)

# Save non-English rows for manual review
print(f"Saving non-English rows to {non_english_file_path_xlsx}...")
non_english_data.to_excel(non_english_file_path_xlsx, index=False)

# 4. Clean the text for English rows only
def clean_text(text):
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    text = text.lower()                              # Convert to lowercase
    return text.strip()

tqdm.pandas(desc="Cleaning text")
english_data['Cleaned_Body'] = english_data['Body_Expanded'].progress_apply(clean_text)


# 5. Tokenization and lemmatization with SpaCy
def process_text_spacy(text):
    doc = nlp(text)
    # Lemmatize, remove stopwords, and keep only alphabetic tokens
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

tqdm.pandas(desc="Tokenizing and lemmatizing")
english_data['Lemmatized_Tokens'] = english_data['Cleaned_Body'].progress_apply(process_text_spacy)

# 6. Prepare text for TF-IDF by joining tokens
print("Preparing text for TF-IDF...")
english_data['Processed_Text'] = english_data['Lemmatized_Tokens'].apply(lambda tokens: ' '.join(tokens))

# 7. Generate TF-IDF matrix
print("Generating TF-IDF matrix...")
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(english_data['Processed_Text'])

# Convert TF-IDF matrix to DataFrame for inspection
print("Converting TF-IDF matrix to DataFrame...")
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Save TF-IDF matrix to Excel
print(f"Saving TF-IDF matrix to {tfidf_output_xlsx}...")
tfidf_df.to_excel(tfidf_output_xlsx, index=False)

# 8. Save the preprocessed English data
print(f"Saving preprocessed English data to {preprocessed_file_path_csv} and {preprocessed_file_path_xlsx}...")
english_data.to_csv(preprocessed_file_path_csv, index=False)
english_data.to_excel(preprocessed_file_path_xlsx, index=False)

print("Processing complete!")
