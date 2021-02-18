from gensim.models import word2vec
import logging
import numpy as np
import os
import sys
import gensim
import multiprocessing
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from collections import namedtuple
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import itertools

exclude_list =  set(stopwords.words('english'))
word_vec = word2vec.Word2Vec.load("./Models/word2vec.model")
print(word_vec.most_similar('left', topn=10))


commonBiPairs = pickle.load( open( "./Models/commonBiPairs.pkl", "rb" ) )
#modified domainMap
def modify_key(str1):
    #outputstr = preprocess(str1)
    outputstr = str1.replace('-', ' ')
    stemmer = SnowballStemmer("english")
    #print(outputstr)
    #print exclude_list         
    words = outputstr.split(' ')
    words = [stemmer.stem(str(word)) for word in words]
    newContent = ' '.join([word for word in words if word not in exclude_list])
    return(newContent.lstrip().rstrip())

BIRADS_dic = pd.read_csv('./Dictionary/BIRADS_terms.csv')
BIRADS_dic = BIRADS_dic.fillna('N/A')
Dkey = []
for i in range(BIRADS_dic.shape[0]):
    key = BIRADS_dic.iloc[i]['Birads key']
    #key = noteprocessing(key);
    key = modify_key(key)
    for pair in commonBiPairs.keys():
        p = commonBiPairs[pair].split(' ')
        if p[0].islower() and p[1].islower():
            key = key.replace(commonBiPairs[pair], str(p[0]) + '_' + str(p[1]))
    Dkey.append(key)
           
def remove_duplicates(l):
    return list(set(l))
           
Dkey = remove_duplicates(Dkey)
           #common keys
#compute the word map based on common-term dictionary
#for each keyword find the context

def compute_contextvec(tokenized, word_vec):
    Count_Row1 = len(tokenized)
    unWantedWords = ['dot', 'col']
    #Count_Row1 = 1
    token_vec = []
    #for training set
    m_token = []
    for i in range(Count_Row1):
    #domainkey
        tokens_key = []
        RD1 = [x for x in tokenized[i] if x not in unWantedWords]
        
        p = 0
        for k in range(len(RD1)):
            
            sub = RD1[k]
            for s in Dkey:     
                if sub.lower() in s.lower():
                    # print(sub)
                    tokens_key.append([])
                    if(k-1 > -1):
                        tokens_key[p].append(RD1[k-1])
                    if(k-2 > -1):
                        tokens_key[p].append(RD1[k-2])
                    #if(k-3 > -1):
                    #    tokens_key[p].append(tokenized[i][k-3])
                    tokens_key[p].append(RD1[k]) 
                    if(k+1 < len(RD1)):
                        tokens_key[p].append(RD1[k+1])
                    if(k+2 < len(RD1)):
                        tokens_key[p].append(RD1[k+2])
                    #if(k+3 < len(tokenized[i])):
                    #    tokens_key[i][p].append(RD1[i][k+3])
                    p = p+1;
            for s in Gkey:
                if sub.lower() in s.lower(): 
                   # print(sub)
                    tokens_key.append([])
                    if(k-1 > -1):
                        tokens_key[p].append(tokenized[i][k-1])
                    if(k-2 > -1):
                        tokens_key[p].append(tokenized[i][k-2])
                    tokens_key[p].append(sub)
                    if(k+1 < len(RD1)):
                        tokens_key[p].append(RD1[k+1])
                    if(k+2 < len(RD1)):
                        tokens_key[p].append(RD1[k+2])
                    p= p+1
        m_token_temp = unique_nestedlist(tokens_key)
        if(len(m_token_temp)> 0):
            m_token_temp = list(itertools.chain.from_iterable(m_token_temp))
        else:
            m_token_temp = [' ']
        m_token.append(m_token_temp)   
    m_vector = context_averaging_list(word_vec.wv,m_token)
    return m_vector
Gkey = [u'risk', u'pt', u'negex',u'fam', u'prev', u'hx', u'screen', ]
Gkey = Gkey[:-1]
def unique_nestedlist (lst):
    unique = []
    for item in lst:
        if sorted(item) not in unique:
            unique.append(sorted(item))
            
    return unique

#word average approach


def word_contexting(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(200,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def context_averaging_list(wv, text_list):
    return np.vstack([word_contexting(wv, review) for review in text_list])

#text tokenizing
def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


# Transform data (you can add more data preprocessing steps) 

def tokenization(df_temp):
    test_tokenized = []
    for i in range(df_temp.shape[0]):
        newContent = df_temp.iloc[i]['Modified_report']
        words = newContent.split(' ')
        newContent = ' '.join([word for word in words if word not in exclude_list])
        test_tokenized.append(w2v_tokenize_text(newContent))

    X_test_word_average = compute_contextvec(test_tokenized, word_vec)
    return X_test_word_average



