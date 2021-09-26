import re

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import nltk

tokenizer = RegexpTokenizer(r'\w+')
nl_stop = get_stop_words('dutch')
p_stemmer = PorterStemmer()


def remove_metadata(case):
    return case.split('\n')[2]


def subject_and_case(case):
    return (case.split('\n')[0], case.split('\n')[2])


def subject_case_articles(case):
    return case.split('\n')[0], case.split('\n')[2], case.split('\n')[3]


def subject_case_summary(case):
    return case.split('\n')[0], case.split('\n')[2], case.split('\n')[1]


def tokenize_to_words(case):
    return tokenizer.tokenize(re.sub(' +', ' ', re.sub('\t', ' ', case)))


# def removing_non_words(case):
#     return [ i for i in case if i in words ]

def stemming(case):
    return [p_stemmer.stem(i) for i in case]


def remove_stopwords(case):
    return [i for i in case if not i in nl_stop]


def remove_uncommon_words(tokens):
    freq_dist = nltk.FreqDist(tokens)
    rarewords = list(freq_dist.keys())[-5:]
    return [word for word in tokens if word not in rarewords]


def preprocess(case):
    intermediate = case.lower()
    intermediate = tokenize_to_words(intermediate)
    # intermediate = removing_non_words(intermediate)
    intermediate = stemming(intermediate)
    intermediate = remove_stopwords(intermediate)
    return remove_uncommon_words(intermediate)
