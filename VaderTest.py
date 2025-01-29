import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure necessary NLTK resources are downloaded
nltk.download('vader_lexicon')


# Input sentence
sentence = 'Hi Altair, I am encountering an issue with AI Studio 2024.1 (RapidMiner) on my MacBook Air (macOS 10.15.7). Despite following the download instructions precisely, the application does not open. When I attempt to launch it, the icon bounces as though it is about to start, but the app never opens. I ensured no special security settings were blocking the application, and my Analytics professor, who is familiar with this platform, suggested it might be a glitch and advised me to contact Altair support. This is my fifth time reaching out regarding this issue, and I would appreciate your guidance on how to resolve it. Thank you in advance for your assistance.'

# Tokenize the sentence
tokenized_sentence = word_tokenize(sentence)

# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Lists to store words by sentiment
pos_word_list = []
neu_word_list = []
neg_word_list = []

# Analyze each word's sentiment
for word in tokenized_sentence:
    # Get the sentiment score for the word
    score = sia.polarity_scores(word)['compound']

    if score >= 0.05:
        pos_word_list.append(word)
    elif score <= -0.05:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)

# Print results
print('Positive:', pos_word_list)
print('Neutral:', neu_word_list)
print('Negative:', neg_word_list)

# Analyze overall sentence sentiment
score = sia.polarity_scores(sentence)
print('\nScores:', score)
