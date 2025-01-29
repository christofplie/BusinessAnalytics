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

# Flatten and filter tokens (exclude 1-2 character tokens)
all_tokens = [token for sublist in data['Lemmatized_Tokens'].dropna()
             for token in sublist if len(token) > 2]

# Create frequency dictionary
word_freq = pd.Series(all_tokens).value_counts().to_dict()

# Generate word cloud with PowerPoint-optimized dimensions
wordcloud = WordCloud(
    width=1920,
    height=1080,
    background_color='white',
    colormap='viridis',
    max_words=150,
    collocations=False
).generate_from_frequencies(word_freq)

# Create figure with PowerPoint slide dimensions (13.3x7.5 inches for 16:9)
plt.figure(figsize=(19.2, 10.8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Save as high-quality PNG for PowerPoint
wordcloud_path = os.path.join(downloads_folder, "ppt_wordcloud.png")
plt.savefig(wordcloud_path, bbox_inches='tight', dpi=100)
plt.show()
print(f"PowerPoint-ready word cloud saved to: {wordcloud_path}")
