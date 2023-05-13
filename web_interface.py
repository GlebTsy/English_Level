import streamlit as st
import pandas as pd
import re
import numpy as np
import pysrt
import spacy
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from pycaret.classification import load_model, predict_model


TITLE = 'Language level prediction'
st.set_page_config(
                   page_title=TITLE,
                   page_icon='🎬',
                   initial_sidebar_state='expanded')
st.title(TITLE)
st.write('This ML-based app predicts the linguistic level of a film for English learners. The classification is based on CEFR levels (A1, A2, B1, B2, C1, C2). Upload susbtitles in .srt format to know the level.')

model = load_model('language_model')

upload_file = st.file_uploader('Upload subtitles in .srt format', type='srt')

nlp = spacy.load("en_core_web_sm")
vectorizer = CountVectorizer(max_features=4000, stop_words=stopwords.words('english'))
tfidfconverter = TfidfTransformer()

def predictions(model, upload_file):
    try:
        subs = pysrt.from_string(upload_file.getvalue().decode('cp1252'))
        print('Decode ANSI success')
        if subs.text == '':
            subs = pysrt.from_string(upload_file.getvalue().decode('utf-16'))
            print('Decode UTF-16 success')
        print('Read file success')
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return
    
    text = re.sub('<i>|</i>', '', subs.text)
    text = re.sub('\<.*?\>', '', text)      
    text = re.sub('\n', ' ', text)               
    text = re.sub('\(.*?\)', '', text)           
    text = re.sub('\[.*?\]', '', text)           
    text = re.sub('[A-Z]+?:', '', text)          
    text = re.sub('\.+?:', '\.', text)           
    text = text.lower()
    text = re.sub('[^a-z\.\!\?]', ' ', text)     
    text = re.sub(' +', ' ', text)               
    spacy_results = nlp(text)
    text = ' '.join([token.lemma_ for token in spacy_results])
    text = [text]
    text = vectorizer.fit_transform(text).toarray()
    text = tfidfconverter.fit_transform(text).toarray()

    N = 0
    if text.shape[1] < 4000:
        N = 4000 - text.shape[1]
        text = np.pad(text, ((0, 0), (0, N)), 'constant')
    else:
        pass
    
    col = []
    for i in range(1,4001):
        col.append('feature_'+str(i))

    df = pd.DataFrame(text, columns=col)
    
    predictions = predict_model(model, data = df)['prediction_label'][0]
    return predictions


if upload_file:

    print(upload_file.name)
    
    st.header(f'This film is labeled **:[{predictions(model, upload_file)}]** on CEFR classification')