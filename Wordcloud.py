import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from ast import literal_eval

# File paths
downloads_folder = os.path.expanduser("~/Downloads")
input_file = os.path.join(downloads_folder, "latest_posts_cleaned.csv")

# Load data
data = pd.read_csv(input_file)

# Convert stringified lists to actual lists of tokens
data['Lemmatized_Tokens'] = data['Lemmatized_Tokens'].apply(literal_eval)

# Flatten all tokens into a single list
all_tokens = [token for sublist in data['Lemmatized_Tokens'].dropna() for token in sublist]

# Create frequency dictionary
word_freq = pd.Series(all_tokens).value_counts().to_dict()

# Generate word cloud
wordcloud = WordCloud(
    width=1200,
    height=800,
    background_color='white',
    colormap='viridis',
    max_words=150
).generate_from_frequencies(word_freq)

# Plot
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Save and show
wordcloud_path = os.path.join(downloads_folder, "lemmatized_wordcloud.png")
plt.savefig(wordcloud_path, bbox_inches='tight')
plt.show()
print(f"Word cloud saved to: {wordcloud_path}")

