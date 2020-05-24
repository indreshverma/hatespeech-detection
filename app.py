#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['processed_tweet']
        data = [message]
        vect = cv.transform(data)
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction='Hate speech detection label is {}'.format(my_prediction[0]))



if __name__ == '__main__':
	app.run()
