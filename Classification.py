import pickle
import pandas as pd
import re #regexes
import sys #command line arguments
import os, os.path
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.collocations import *
from nltk import word_tokenize
from gensim.models import word2vec
import os
import sys
import gensim
import multiprocessing
from sklearn.model_selection import train_test_split
from collections import namedtuple
import logging
import pickle


def classificationReport(X_test_word_average):
    classifier_U = pickle.load(open( "./Models/classifier_U.pkl", "rb" ) )
    y_pred = classifier_U.predict(X_test_word_average)
    return y_pred
