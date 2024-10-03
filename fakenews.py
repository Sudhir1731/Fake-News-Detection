import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
nltk.download('stopwords')

news_df = pd.read_csv('train.csv')
news_df = news_df.fillna('')
news_df['content'] = news_df['author']+' '+news_df['title']
x = news_df.drop('label',axis=1)
y = news_df['label']

ps = PorterStemmer()
def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]'," ",content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = " ".join(stemmed_content)
  return stemmed_content


news_df['content'] = news_df['content'].apply(stemming)

x = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

model = LogisticRegression()
model.fit(x_train,y_train)



#website
st.title('Fake News Detector')
input_text = st.text_input("Enter news article")

def prediction(input_text):
  input_data = vector.transform([input_text])
  prediction = model.predict(input_data)
  return prediction [0]

if input_text:
  pred = prediction(input_text)
  if pred == 1:
    st.write('The News is Fake')
  else :
    st.write('The News is Real')

