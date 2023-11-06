""" 
Utility functions for interacting with the database.
Including:
1. connect to the database
2. Create Table kamus alay & Abusive
3. Insert Result of data cleansing 
"""

import pandas as pd
import sqlite3

def create_connection():
    conn = sqlite3.connect('platinum_challenge.db')
    return conn

def insert_dictionary_to_db(conn):
    abusive_csv_file = "csv_data/abusive.csv"
    alay_csv_file = "csv_data/alay.csv"
    clean_data_file = "csv_data/cleaned_text.csv"

    # Read csv file to dataframe
    print("Reading csv file to dataframe...")
    df_abusive = pd.read_csv(abusive_csv_file)
    df_alay = pd.read_csv(alay_csv_file, encoding="latin-1")
    df_clean_data = pd.read_csv(clean_data_file, sep="\t")
    df_clean_data.drop('Unnamed: 0', axis=1, inplace=True)

    # Standardize column name
    df_abusive.columns = ['word']
    df_alay.columns = ['alay_word', 'formal_word']
    #df_clean_data.columns =['kalimat', 'sentiment', 'clean text', 'clean abusive', 'clean alay', 'clean abusive alay']
  
    # Insert dataframe to database
    print("Inserting dataframe to database...")
    df_abusive.to_sql('abusive', conn, if_exists='replace', index=False)
    df_alay.to_sql('alay', conn, if_exists='replace', index=False)
    df_clean_data.to_sql('clean data', conn, if_exists='replace', index=False)
    print("Inserting dataframe to database success!")

def insert_result_to_db(conn, raw_text, clean_text, sentiment_result):
    # Insert result to database
    print("Inserting result to database...")
    df = pd.DataFrame({'raw_text': [raw_text], 'clean_text': [clean_text], 'sentiment': [sentiment_result]})
    df.to_sql('neural_network_result', conn, if_exists='append', index=False)
    print("Inserting result to database success!")

def insert_upload_result_to_db(conn, result_df):
    # Insert result to database
    print("Inserting result to database...")
    result_df.to_sql('neural_network_result', conn, if_exists='append', index=False)
    print("Inserting result to database success!")

def show_sentiment_result(conn):
    # Show cleansing result
    print("Showing sentiment analysis result...")
    df = pd.read_sql_query("SELECT * FROM neural_network_result", conn)
    return df.T.to_dict()

def show_sentiment_result_LSTM(conn):
    # Show cleansing result
    print("Showing sentiment analysis result...")
    df = pd.read_sql_query("SELECT * FROM LSTM_result", conn)
    return df.T.to_dict()

def insert_result_to_db_LSTM(conn, raw_text, clean_text, sentiment_result):
    # Insert result to database
    print("Inserting result to database...")
    df = pd.DataFrame({'raw_text': [raw_text], 'clean_text': [clean_text], 'sentiment': [sentiment_result]})
    df.to_sql('LSTM_result', conn, if_exists='append', index=False)
    print("Inserting result to database success!")

def insert_upload_result_deep_learning_to_db(conn, sentiment):
    # Insert result to database
    print("Inserting sentiment to database...")
    sentiment.to_sql('LSTM_result', conn, if_exists='append', index=False)
    print("Inserting sentiment to database success!")

