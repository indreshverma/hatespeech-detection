#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:35:28 2020

@author: devesh
"""
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   

from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
#df = pd.read_csv('train.csv')
#
#df.isnull().sum()
#df['label']=df['toxic']+df['severe_toxic']+ df['obscene']+df['threat']+df['insult']+df['identity_hate']
#df['label'] = np.where(df['label']==0,0,1)
#df.columns
#df.drop(['id','toxic','severe_toxic','obscene', 'threat','insult','identity_hate'], axis=1,inplace=True)
#df['cleaned_tweet'] = df.comment_text.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
#df['cleaned_tweet'] = df.comment_text.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
#
#
#def clean_text(text):
#    
#    ## Remove puncuation
#    text = text.translate(string.punctuation)
#    
#    ## Convert words to lower case and split them
#    text = text.lower().split()
#    
#    ## Remove stop words
#    stops = set(stopwords.words("english"))
#    text = [w for w in text if not w in stops and len(w) >= 3]
#    
#    text = " ".join(text)
#    ## Clean the text
#    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
#    text = re.sub(r"what's", "what is ", text)
#    text = re.sub(r"\'s", " ", text)
#    text = re.sub(r"\'ve", " have ", text)
#    text = re.sub(r"n't", " not ", text)
#    text = re.sub(r"i'm", "i am ", text)
#    text = re.sub(r"\'re", " are ", text)
#    text = re.sub(r"\'d", " would ", text)
#    text = re.sub(r"\'ll", " will ", text)
#    text = re.sub(r",", " ", text)
#    text = re.sub(r"\.", " ", text)
#    text = re.sub(r"!", " ! ", text)
#    text = re.sub(r"\/", " ", text)
#    text = re.sub(r"\^", " ^ ", text)
#    text = re.sub(r"\+", " + ", text)
#    text = re.sub(r"\-", " - ", text)
#    text = re.sub(r"\=", " = ", text)
#    text = re.sub(r"'", " ", text)
#    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
#    text = re.sub(r":", " : ", text)
#    text = re.sub(r" e g ", " eg ", text)
#    text = re.sub(r" b g ", " bg ", text)
#    text = re.sub(r" u s ", " american ", text)
#    text = re.sub(r"\0s", "0", text)
#    text = re.sub(r" 9 11 ", "911", text)
#    text = re.sub(r"e - mail", "email", text)
#    text = re.sub(r"j k", "jk", text)
#    text = re.sub(r"\s{2,}", " ", text)
#    ## Stemming
#    text = text.split()
#    stemmer = SnowballStemmer('english')
#    stemmed_words = [stemmer.stem(word) for word in text]
#    text = " ".join(stemmed_words)
#    return text
#
#df['processed_tweet'] = df['cleaned_tweet'].apply(clean_text)
#df.columns
#df.drop(['comment_text','cleaned_tweet'], axis=1, inplace=True)
#
##df.to_csv('data.csv')
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))


lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
data=pd.read_csv('train.csv')
data.info

data.isnull().sum()
data['label']=data['toxic']+data['severe_toxic']+ data['obscene']+data['threat']+data['insult']+data['identity_hate']
data.head(7)

data['label']= np.where(data['label']==0,0,1)
sns.countplot(x='label',data=data)
check=pd.crosstab(index=data["label"],columns="count")
check
data1=data.iloc[data['label'].values==1]
data2=data.iloc[data['label'].values==0]
data_sample=data2.sample(n=64900)

df=pd.concat([data_sample,data1])
df.reset_index(drop=True)
df=df.sample(frac=1).reset_index(drop=True)
df.head(20)
datafinal=df.iloc[:,[1,8]]

datafinal['cleaned_tweet'] = datafinal.comment_text.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
datafinal['cleaned_tweet'] = datafinal.comment_text.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))




def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

datafinal['processed_tweet'] = datafinal['cleaned_tweet'].apply(clean_text)


X=datafinal['processed_tweet']
idf=cv.idf_
cv = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    
    ngram_range=(1,2),
    max_features=30000)

# fit and transform on it the training features
X=cv.fit_transform(datafinal['processed_tweet'])
pickle.dump(cv, open('transform.pkl','wb'))

dic=dict(zip(cv.get_feature_names(),idf))
print(dic)

X.shape
Y=datafinal.label

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=13)

classifier = LogisticRegression()
classifier.fit(X_train,y_train)

Y_Pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, Y_Pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, Y_Pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, Y_Pred))

from sklearn import model_selection, naive_bayes, svm
from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV

parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.0001,0.001,0.0011,0.0012,0.0013,0.0014,0.0015,0.01,0.1], 'kernel': ['rbf']},
 ]

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced')
SVM.fit(X_train,y_train)
predictions_SVM = SVM.predict(X_test)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)
print(classification_report(y_test, predictions_SVM))
confusion_matrix(y_test, predictions_SVM)

datafinal['processed_tweet'][14]
datafinal.label.head(40)
test='you deserve to be ashamed'

test=clean_text(test)
test=[test]
print(SVM.predict(cv.transform(test)))
print(cv.get_feature_names)

filename="nlp_model.pkl"

pickle.dump(SVM, open(filename,'wb'))