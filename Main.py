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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from Cleaning import noteprocessing
from Vectorization import tokenization
from Classification import classificationReport

## I/O
name = input("Enter file path + file name ")


## check file read
try:
    df_report = pd.read_excel(name)
except:
    print('Cannot read file')
    sys.exit(0)

df_temp = pd.DataFrame(columns=('report_id', 'Modified_report'))
Count_Row = df_report.shape[0]
for i in range(Count_Row):
    try:
        txt = str(df_report.iloc[i]['Report']).lower() #taking only the finding
        report = txt.split('impression:')
        modifiedContent = noteprocessing(report[0]);
        df_temp.loc[i] = [df_report.iloc[i][0], modifiedContent]
    except:
        print('Issue with colum name; please provide Report column')
        sys.exit(0)

X_test_word_average = tokenization(df_temp)

df_temp['BIRAD Prediction'] = classificationReport(X_test_word_average)
df_temp['Report'] = list(df_report['Report'])
        
df_temp.to_excel('./Output/annotated_BIRADS.xlsx')
