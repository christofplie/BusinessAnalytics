import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (if not already done)
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
input_file_path = os.path.join(downloads_folder, "latest_posts.csv")
preprocessed_file_path_csv = os.path.join(downloads_folder, "latest_posts_cleaned.csv")
preprocessed_file_path_excel = os.path.join(downloads_folder, "latest_posts_cleaned.xlsx")
preprocessed_subset_file_path_excel = os.path.join(downloads_folder, "latest_posts_cleaned_subset.xlsx")

# 1. Load the dataset
data = pd.read_csv(input_file_path)

# 2. Handle missing values
data = data.dropna(subset=["Body"]).reset_index(drop=True)

# 3. Clean the text
def clean_text(text):
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\W+', ' ', text)    # Remove special characters
    text = text.lower()                 # Convert to lowercase
    return text

data['Cleaned_Body'] = data['Body'].apply(clean_text)

# 4. Tokenize and remove stop words
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

data['Tokens'] = data['Cleaned_Body'].apply(tokenize_and_remove_stopwords)

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

data['Lemmatized_Tokens'] = data['Tokens'].apply(lemmatize_words)

# 6. Generate TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=1000)  # Limit to top 1000 terms
tfidf_matrix = vectorizer.fit_transform(data['Cleaned_Body'])

# Convert TF-IDF matrix to a DataFrame for inspection
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# 7. Save preprocessed data for further analysis
data.to_csv(preprocessed_file_path_csv, index=False)
data.to_excel(preprocessed_file_path_excel, index=False)

print(f"Preprocessed data saved to {preprocessed_file_path_csv}")
