# Create Function to obtain sentiment, consine similarity and topic
import pandas as pd
import numpy as np
import re
import gensim
import string 
import nltk
import pickle

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models  

def topic_modelling_dictionary(corpus_filepath = r"./Topic Modelling/dictionary_bigram_cleaned_reviews_nverbs_only.txt"):
    text_data = []
    
    with open(corpus_filepath, 'r') as f:
        for line in f:
            tokens = word_tokenize(line)
            text_data.append(tokens)

    dictionary = gensim.corpora.Dictionary(text_data)

    return dictionary


def preprocess(x):

    # Instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # 1. Lower case
    tmp = x.lower()
    
    # 2. Tokenize the sentences
    tokens = word_tokenize(tmp)

    # Stopword List
    stopword_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', \
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', \
        'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', \
        'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',\
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', \
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', \
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',\
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',\
        'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", \
        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', \
        "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'wine', 'drink', 'shows', 'also', 'made', 'like', 'bit', 'give', 'opens', 'alongside', \
        'along', 'ready', 'yet', 'one', 'feels', 'almost']

    # 3. Remove stopwords
    no_stopwords = [word for word in tokens if word not in stopword_list and word.isalpha()]
    
    # 4. Lemmatize
    lemma_text = ' '.join([lemmatizer.lemmatize(word) for word in no_stopwords])

    # 5. Remove punctuations
    processed_text = lemma_text.translate(str.maketrans('', '', string.punctuation))
        
    return processed_text


# Create Function to obtain Topic Number
def assign_topic(x):
    if x == 0:
        return "Red Wine"
    elif x == 1:
        return "Cherry Plum Wine"
    elif x == 2:
        return "Champagne"
    elif x == 3:
        return "Berries Wine"
    elif x == 4:
        return "Cherry Plum Wine"
    elif x == 5:
        return "White Wine"
    else:
        return "Peach Wine" 


def topic_modelling(review, dict_corpus_path = r"./Topic Modelling/dictionary_bigram_cleaned_reviews_nverbs_only.txt", lda_model_path = r"/Topic Modelling/LDA Gensim (Initial Model)/LDA Gensim (Initial Model)/lda_tfidf_bigram_full_model.pk"):
# def topic_modelling(review, dict_corpus_path, lda_model_path = "./LDA Gensim (Initial Model)/lda_tfidf_bigram_full_model.pk'"):
    
    test_data_cleaned = preprocess(review)
    test_data_cleaned_tokens = word_tokenize(test_data_cleaned)
    test_data_cleaned_tokens_bow = topic_modelling_dictionary(dict_corpus_path).doc2bow(test_data_cleaned_tokens)

    # Reload LDA BIGRAM TF-IDF Model
    with open(lda_model_path, 'rb') as file:
        lda_model = pickle.load(file)

    topics = lda_model.get_document_topics(test_data_cleaned_tokens_bow)
    sorted_topics = sorted(topics, key=lambda x:x[1],reverse=True)
    sorted_topic_top = sorted_topics[0][0]

    return assign_topic(sorted_topic_top)


review = "Delivering profound notes of black and red currants, blackberry fruit, blood orange citrus, and dried raspberries underscored by baking spices \
            and dried red florals, this pinot noir is also a textural masterpiece with mouthwatering acidity and grippy cedar-like tannins"


if __name__ == "__main__":
    print("Topic: ", topic_modelling(review, r"./Topic Modelling/dictionary_bigram_cleaned_reviews_nverbs_only.txt", r"./Topic Modelling/LDA Gensim (Initial Model)/lda_tfidf_bigram_full_model.pk"))
