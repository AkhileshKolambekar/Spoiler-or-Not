import streamlit as st
import pickle
import string
import nltk
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import html5lib
import requests
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer

st.title('Spoiler or Not :film_projector:')

#Load the model
model = pickle.load(open("lr.pkl","rb"))

#Load the column transformer
ct = pickle.load(open('ct.pkl','rb'))

#Function to remove punctuations
def remove_punc(txt):
    new_txt = ""
    punctuation = string.punctuation
    for letter in txt:
        if letter.isdigit() == False:
            if letter == '.' or letter == '*':
                new_txt += ''
            elif letter not in punctuation and letter != '.':
                new_txt += letter.lower()
    return new_txt

#Function to remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokens):
    new_txt = ""
    for word in tokens.split():
        if word not in stopwords:
            new_txt += word +" "
    return new_txt

#Function to lemmatize text
Lem = nltk.WordNetLemmatizer()
def lemmatize_text(tokens):
    new_txt = ""
    for word in tokens.split():
        new_txt += Lem.lemmatize(word) + " "
    return new_txt

#Function to get the plot of the movie from web
def get_plot(movie):
    title = ''
    for word in movie.split():
        title += word + '+'
    google_link = 'https://www.google.com/search?q='+title[:-1]
    r1 = requests.get(google_link)
    soup1 = BeautifulSoup(r1.text,'html5lib')
    wiki = soup1.find('div',attrs = {'class':"Gx5Zad fP1Qef xpd EtOod pkphOe"})
    link = wiki.find_next('a')['href'][7:]
    if link.rfind(')') != -1:
        r = requests.get(link[7:link.rfind(')')+1])
    else:
        title = ''
        for word in movie.split():
            title += word + '_'
        r = requests.get('https://en.wikipedia.org/wiki/'+title[:-1])
    soup = BeautifulSoup(r.text,'html5lib')
    
    plot = ""
    span = soup.select('#Plot')[0]
    span_srcline = span.sourceline
    span_srcpos = span.sourcepos
    for tag in span.find_all_next("p"):
        if tag.sourceline - span_srcline<10:
            plot += f'{tag}'
    output_string = re.sub(r'<[^>]*>', '', plot)
    return output_string

#Taking input from the user
movie_name = st.text_input('Enter movie name')
review = st.text_input('Enter your review')

if st.button('Check'):
        if len(movie_name)==0:
            st.error('Please enter movie name')
        elif len(review) == 0:
            st.error('Please enter the review')
        else:
            plot = get_plot(movie_name) 

            data = pd.DataFrame({'review_text':[review],'plot_synopsis':[plot]})

            data['plot_synopsis'] = data['plot_synopsis'].apply(lambda x: remove_punc(x))
            data['plot_synopsis'] = data['plot_synopsis'].apply(lambda x: remove_stopwords(x))
            data['plot_synopsis'] = data['plot_synopsis'].apply(lambda x: lemmatize_text(x))

            data['review_text'] = data['review_text'].apply(lambda x: remove_punc(x))
            data['review_text'] = data['review_text'].apply(lambda x: remove_stopwords(x))
            data['review_text'] = data['review_text'].apply(lambda x: lemmatize_text(x))

            transformed_data = ct.transform(data)

            prediction = model.predict(transformed_data)
            if prediction[0] == 0:
                st.write("The review is spoiler free")
            else:
                st.write("The review contains spoilers")