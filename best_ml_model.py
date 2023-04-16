import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Clean emojis from text

def strip_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)


# read data

df_train = pd.read_csv("Corona_NLP_train.csv", 
                       encoding='latin-1')
df_test = pd.read_csv("Corona_NLP_test.csv", 
                      encoding='latin-1')

df_train = df_train[['OriginalTweet', 'Sentiment']]
df_test = df_test[['OriginalTweet', 'Sentiment']]

# clean data
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(strip_emoji)
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(strip_all_entities)
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(clean_hashtags)
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(filter_chars)
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(remove_mult_spaces)

df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(strip_emoji)
df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(strip_all_entities)
df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(clean_hashtags)
df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(filter_chars)
df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(remove_mult_spaces)

# Preprocess the data
df_train['Sentiment'] = df_train['Sentiment'].replace(['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'], 
                                                      [0, 1, 2, 3, 4])
X = df_train['OriginalTweet'].values
y = df_train['Sentiment'].values

# Create a CountVectorizer representation of the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
X = X.astype(np.float64)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(C = 1, max_iter=1000, penalty='l1', solver='saga', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f1)