# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:45:33 2019

@author: S534735
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_sentiments(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def split_sentiments(sentiments):
    xs = [sentiment['neg'] for sentiment in sentiments]
    ys = [sentiment['neu'] for sentiment in sentiments]
    zs = [sentiment['pos'] for sentiment in sentiments]
    return xs,ys,zs

# lexical Diversity
def lexical_diversity(text):
    tokens = text.lower().split()
    return len(set(tokens))/len(tokens)

def average_words(statuses):
    total_words = sum([ len(s.split()) for s in statuses ]) 
    return total_words/len(statuses)