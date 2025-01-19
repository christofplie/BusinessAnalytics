import os
import re
import pandas as pd
import spacy
import contractions
from tqdm import tqdm
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Fix randomness in langdetect
DetectorFactory.seed = 0

# Load SpaCy model for English
nlp = spacy.load("en_core_web_sm")

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
input_file_path = os.path.join(downloads_folder, "rapidminernew_cleaned.csv")
preprocessed_file_path_csv = os.path.join(downloads_folder, "latest_posts_cleaned_new.csv")
preprocessed_file_path_xlsx = os.path.join(downloads_folder, "latest_posts_cleaned_new.xlsx")
non_english_file_path_xlsx = os.path.join(downloads_folder, "non_english_posts_new.xlsx")

# 1. Load the dataset
print("Loading dataset...")
data = pd.read_csv(input_file_path)
data = data.dropna(subset=["body"]).reset_index(drop=True)

# 2. Expand contractions
print("Expanding contractions...")
def expand_contractions(text):
    return contractions.fix(text)

data['Body_Expanded'] = data['body'].apply(expand_contractions)

# 3. Detect and filter by language
print("Detecting language...")
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

tqdm.pandas(desc="Detecting language")
data['Language'] = data['Body_Expanded'].progress_apply(detect_language)

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
    return text.lower().strip()          # Convert to lowercase

tqdm.pandas(desc="Cleaning text")
english_data['Cleaned_Body'] = english_data['Body_Expanded'].progress_apply(clean_text)

# 5. Remove specific unwanted words
def remove_unwanted_words(text, words_to_remove):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')\b'
    cleaned_text = re.sub(pattern, '', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace

unwanted_words = ["nbsp"]

tqdm.pandas(desc="Removing unwanted words")
english_data['Cleaned_Body'] = english_data['Cleaned_Body'].progress_apply(
    lambda text: remove_unwanted_words(text, unwanted_words)
)

# 6. Tokenization and lemmatization with SpaCy
def process_text_spacy(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

tqdm.pandas(desc="Tokenizing and lemmatizing")
english_data['Lemmatized_Tokens'] = english_data['Cleaned_Body'].progress_apply(process_text_spacy)

# 7. Create a processed text column by joining lemmatized tokens
print("Creating processed text column...")
tqdm.pandas(desc="Joining lemmatized tokens")
english_data['Processed_Text'] = english_data['Lemmatized_Tokens'].progress_apply(lambda tokens: ' '.join(tokens))

# 8. Save preprocessed English data
print(f"Saving preprocessed English data to {preprocessed_file_path_csv} and {preprocessed_file_path_xlsx}...")
english_data.to_csv(preprocessed_file_path_csv, index=False)
english_data.to_excel(preprocessed_file_path_xlsx, index=False)

print("Preprocessing complete!")

