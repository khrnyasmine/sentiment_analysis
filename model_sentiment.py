"""
Function for sentiment analysis
"""

import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Open Neural Network Model
count_vect = pickle.load(open("neural_network_model/feature.p", 'rb'))
with open("neural_network_model/model.p", "rb") as f:
    lr = pickle.load(f)

# Open LSTM resources
file = open("lstm_model/x_pad_sequences.pickle", 'rb')
feature_file_from_LSTM = pickle.load(file)
file.close

file = open("lstm_model/tokenizer.pickle", 'rb')
tokenizer = pickle.load(file)
file.close

model_file_from_LSTM = load_model('lstm_model/model_lstm.h5')

def text_cleansing(text):
    # lowercase
    clean_text = str(text).lower()
    # clean URL
    clean_text = re.sub(r'(http\S+|www\S+)', '', clean_text).strip()
    # clean emoticon byte
    clean_text = clean_text.replace("\\", " ")
    clean_text = re.sub(r'\bx[0-9a-fA-F]{2,}', ' ', clean_text)
    clean_text = re.sub(r'\bn\b', ' ', clean_text)
    clean_text = re.sub('\\+', ' ', clean_text)
    clean_text = re.sub('  +', ' ', clean_text)
    # clean punctuations
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_text)
    # clean username
    clean_text = re.sub(r'\buser\b', '', clean_text, flags=re.IGNORECASE)
    # clean rt (retweet)
    clean_text = re.sub(r'\brt\b', '', clean_text, flags=re.IGNORECASE)
    return clean_text

# Neural Network Sentiment Analysis
def neural_sentiment(text):
    # Clean Text
    clean_text = text_cleansing(text)

    # Sentiment analysis
    text_vect = count_vect.transform([clean_text])
    sentiment_result = lr.predict(text_vect)[0]
    return sentiment_result

def neural_files(file_upload):
    # Get only the first column
    df_upload = pd.DataFrame(file_upload.iloc[:,0])
    
    # Rename column to "raw_text"
    df_upload.columns = ["raw_text"]

    # Clean text with text_cleansing function
    # Save to "clean_text" column
    df_upload["clean_text"] = df_upload["raw_text"].apply(text_cleansing)
    
    # Neural network sentiment analysis
    df_upload["sentiment"] = df_upload["clean_text"].apply(neural_sentiment)
    print("Neural network sentiment analysis success!")
    return df_upload

def deep_learning(clean_text):
    sentiment = ['negative', 'neutral', 'positive']
    clean_text = [clean_text]

    feature = tokenizer.texts_to_sequences(clean_text)
    feature = pad_sequences(feature, maxlen=feature_file_from_LSTM.shape[1])

    prediction = model_file_from_LSTM.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    return get_sentiment

def deep_learning_upload(file_upload):
    df_upload = pd.DataFrame(file_upload.iloc[:,0])

    df_upload.columns = ["raw_text"]

    df_upload["clean_text"] = df_upload["raw_text"].apply(text_cleansing)
    df_upload["sentiment"] = df_upload["clean_text"].apply(deep_learning)
    print("Cleansing text success!")
    return df_upload