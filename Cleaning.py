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


commonBiPairs = pickle.load( open( "./Models/commonBiPairs.pkl", "rb" ) )
#read domain and common dictionary
#compute the word map based on common-term dictionary
GeneralwordMap = {}
Gterms = []
with open('./Dictionary/clever_base_terminologyv2.txt', 'r') as termFile:
    for line in termFile:
        words = line.split('|')
        if len(words) == 3:
            GeneralwordMap[' ' + words[1].lstrip(' ').rstrip(' ') + ' '] = ' ' + words[2].replace('\n', '').lstrip(' ').rstrip(' ') + ' '
            Gterms.append(' ' + words[1].lstrip(' ').rstrip(' ') + ' ')
Gterms.sort(key = len)
Gterms.reverse()


domain_dic = pd.read_csv('./Dictionary/DicFinal_IB_SB.csv', sep=',', encoding='latin1')
domain_dic = domain_dic.fillna(value='null')
DomainwordMap = {}
Dterms = []
for i in range(domain_dic.shape[0]):
    current_entry = domain_dic.iloc[i]
    if(current_entry['termSyn']!= 'null'):
        DomainwordMap[' ' + current_entry['termSyn'] + ' '] = ' ' + current_entry['term'] + ' '
        Dterms.append(' ' + current_entry['termSyn'] + ' ')
    if(current_entry['subClass']!= 'null'):
        DomainwordMap[' ' + current_entry['subClass'] + ' '] = ' ' + current_entry['term'] + ' '
        Dterms.append(' ' + current_entry['subClass'] + ' ')
    if(current_entry['subClassSyn']!= 'null'):
        DomainwordMap[' ' + current_entry['subClassSyn'] + ' '] = ' ' + current_entry['term'] + ' '
        Dterms.append(' ' + current_entry['subClassSyn'] + ' ')
Dterms.sort(key = len)
Dterms.reverse()

date_input = "" #global for date input (can be removed if we use lambdas later)
pairs = {} #global variable to count common pairs
exclude_list =  set(stopwords.words('english'))


def concatenate_into_string(infile):
    total_text = ""
    for line in infile:
        line = line.replace('\n', ' ')
        total_text += line
    return total_text


def substitute_word_for_digit(num,join=True):
    '''words = {} convert an integer number into words'''
    units = ['',' one ',' two ',' three ',' four ',' five ',' six ',' seven ',' eight ',' nine ']
    teens = ['',' eleven ',' twelve ',' thirteen ',' fourteen ',' fifteen ',' sixteen ', \
             ' seventeen ',' eighteen ',' nineteen ']
    tens = ['',' ten ',' twenty ',' thirty ',' forty ',' fifty ',' sixty ',' seventy ', \
            ' eighty ',' ninety ']
    thousands = ['',' thousand ',' million ',' billion ', ' trillion ',' quadrillion ', \
                 ' quintillion ',' sextillion ',' septillion ',' octillion ', \
                 ' nonillion ',' decillion ',' undecillion ',' duodecillion ', \
                 ' tredecillion ',' quattuordecillion ',' sexdecillion ', \
                 ' septendecillion ',' octodecillion ',' novemdecillion', \
                 ' vigintillion ']
    words = []
    if int(num)==0: words.append(' zero ')
    if int(num)>=1000: words.append(' ')
    else:
        numStr = num
        numStrLen = len(numStr)
        groups = (numStrLen+2)/3
        groups = int(groups)
        numStr = numStr.zfill(groups*3)
        for i in range(0,groups*3,3):
            h,t,u = int(numStr[i]),int(numStr[i+1]),int(numStr[i+2])
            g = int(groups-(i/3+1))
            if h>=1:
                words.append(units[h])
                words.append(' hundred ')
            if t>1:
                words.append(tens[t])
                if u>=1: words.append(units[u])
            elif t==1:
                if u>=1: words.append(teens[u])
                else: words.append(tens[t])
            else:
                if u>=1: words.append(units[u])
            #if (g>=1) and ((h+t+u)>0): words.append(thousands[g]+',') #replace all thousand w
    if join: return ' '.join(words)
    return words


def remove_forbidden_tokens(output, exclude_list):
    for item in exclude_list:
        output = re.sub(r""+re.escape(item)+r"", " ", output)
    return output



'''
The primary method. Takes input as a string and an exclusion list of strings and outputs a string.
'''
def preprocess(inputstr):
    #output = inputstr.lower() #tolowercase
    re1='(\\d+)'	# Integer Number 1
    re2='(\\/)'	# Any Single Character 1
    re3='(\\d+)'	# Integer Number 2
    re4='(\\/)'	# Any Single Character 2
    re5='(\\d+)'	# Integer Number 3

    date = re.compile(re1+re2+re3+re4+re5,re.IGNORECASE|re.DOTALL)
    output = re.sub(date, ' ', inputstr)
    num = re.findall(r'\d+', output)
   #num = filter(str.isdigit, output)
    for i in num:
        wordnum = substitute_word_for_digit(i) #substitute single digit numbers with words
        output = output.replace(i,wordnum)
        #output = output.replace(i,' ') #try without the number
    #output = remove_forbidden_tokens(output, exclude_list)
    
    #str1 = output.replace('\n', ' ')
    #str1 = str1.replace('\r', ' ')
   
    str1 = output.replace('.', ' ')
    str1 = str1.replace(',', ' ')
    str1 = str1.replace('-', ' dash ')
    str1 = str1.replace(':',' ')
    str1  = str1.replace('%',' PERCENTAGE ')
    #str1 = re.sub(r'[^\w\s]','',str1 )
    str1  = re.sub(r" +", " ", str1 ) #remove extraneous whitespace
    words = str1.split(' ')
    #newContent = ' '.join([word for word in words if word not in exclude_list])
    newContent = ' '.join(words)
    return newContent


def noteprocessing(rawnote):
    
    str1 = rawnote.lower();
    #uncomment for with semantic mapping
    for term in Dterms:
        if term in str1:
            str1 = str1.replace(term, DomainwordMap[term])
            #print(term)
    
    
    for term in Gterms:
        if term in str1:
            str1 = str1.replace(term, GeneralwordMap[term])
            #print(term)
    
    
    str1 = preprocess(str1)
    stemmer = SnowballStemmer("english")        
    words = str1.split(' ')
    words = [stemmer.stem(str(word)) for word in words]
    newContent = ' '.join(words)

    for pair in commonBiPairs.keys():
        p = commonBiPairs[pair].split(' ')
        if p[0].islower() and p[1].islower():
            newContent = newContent.replace(commonBiPairs[pair], ' ' + str(p[0]) + '_' + str(p[1])+ ' ')
    newContent = " ".join(newContent.split())
    return newContent



