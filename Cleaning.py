from collections import Counter
import nltk
import pandas as pd
import numpy as np
import plotly
from plotly import graph_objs
import re
from nltk.corpus import stopwords

tweetsData = pd.read_csv('data_caa_against_20191219-144345 - data_caa_against_20191219-144345.csv')


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

def preproces(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def tokenize(s):
    return tokens_re.findall(s)

tweetsData['tokenized_text']= tweetsData['text'].apply(preproces)

stop_words = set(stopwords.words('english'))
tweetsData['token_texts'] = tweetsData['tokenized_text'].apply(lambda x : [w for w in x if w.lower() not in stop_words])

#function defined to calculate number of occurences of a symbol
def count_occurences(character, word_array):
    counter = 0
    for j, word in enumerate(word_array):
        for char in word:
            if char == character:
                counter += 1
    return counter

tweetsData['no_of_question_marks'] = tweetsData['token_texts'].apply(lambda txt: count_occurences("?", txt))
tweetsData['no_of_exclamation_marks'] = tweetsData['token_texts'].apply(lambda txt: count_occurences("!", txt))
tweetsData['no_of_hashtags'] = tweetsData['token_texts'].apply(lambda txt: count_occurences("#", txt))
tweetsData['no_of_mentions'] = tweetsData['token_texts'].apply(lambda txt: count_occurences("@", txt))

def count_by_regex(regex,plain_text):
    return len(re.findall(regex,plain_text))

#Calculates number of URLs in a tweet
tweetsData['no_of_urls'] = tweetsData['text'].apply(lambda txt:count_by_regex("http.?://[^\s]+[\s]?",txt))


# Function to Remove URLs and mentions from tweets
def remove_url_by_regex(pattern,string):
    return re.sub(pattern,"", string)
#Removes URLs
tweetsData['cleaned_text'] = tweetsData['text'].apply(lambda txt:remove_url_by_regex("http.?://[^\s]+[\s]?",txt))

#Removes mentions
tweetsData['cleaned_text'] = tweetsData['cleaned_text'].apply(lambda txt:remove_url_by_regex(r'(?:@[\w_]+)',txt))

#Calculates number of colon marks
tweetsData['no_of_colon_marks'] = tweetsData['cleaned_text'].apply(lambda txt: count_occurences(":", txt))

#Remove punctuation marks
tweetsData['cleaned_text'] = tweetsData['cleaned_text'].apply(lambda txt:remove_url_by_regex(r'[,|:|\|=|&|;|%|$|@|^|*|-|#|?|!|.]',txt))

#Counts number of words
tweetsData['no_of_words'] = tweetsData['cleaned_text'].apply(lambda txt:len(re.findall(r'\w+',txt)))

tweetsData.to_csv("FinalDS_AdditionalFeatures.csv")

data=pd.read_csv("FinalDS_AdditionalFeatures.csv")
#data.sort_values(["user.followers_count"], axis=0,
                # ascending=True, inplace=True)
"""
#LOGIC FOR FAKE BOT DETECTION!!!!!!

average=data['user.followers_count'].mean()
data.columns =[column.replace(".", "_") for column in data.columns]
filter1 = data["user_followers_count"]<=average
filter2 = data["user_verified"]=="FALSE"
filter3 = data["no_of_urls"]==0
data.where(filter1 & filter2 & filter3, inplace = True)

#data.query('user_followers_count <= average and retweeted == False', inplace = True)
#x=data

print(data)"""