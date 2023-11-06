"""
Flask API Application
"""
from flask import Flask, jsonify, request
import pandas as pd
from time import perf_counter
from flasgger import Swagger, swag_from, LazyString, LazyJSONEncoder
from db import (
    create_connection, insert_dictionary_to_db, 
    insert_result_to_db, show_sentiment_result, show_sentiment_result_LSTM,
    insert_upload_result_to_db, insert_upload_result_deep_learning_to_db, insert_result_to_db_LSTM
)
from model_sentiment import (
    text_cleansing, neural_sentiment,
    neural_files, deep_learning_upload, deep_learning
)

# Prevent sorting keys in JSON response
import flask
flask.json.provider.DefaultJSONProvider.sort_keys = False

# Set Up Database
db_connection = create_connection()
insert_dictionary_to_db(db_connection)
db_connection.close()

# Initialize flask application
app = Flask(__name__)
# Assign LazyJSONEncoder to app.json_encoder for swagger UI
app.json_encoder = LazyJSONEncoder
# Create swagger config & swagger template
swagger_template = {
    "info": {
        "title": LazyString(lambda: "Sentiment Analysis API"),
        "version": LazyString(lambda: "1.0.0"),
        "description": LazyString(lambda: "Dokumentasi API untuk analisis sentimen"),
    },
    "host": LazyString(lambda: request.host)
}
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
    }
# Initialize Swagger from swagger template & config
Swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Homepage
@swag_from('docs/home.yml', methods=['GET'])
@app.route('/', methods=['GET'])
def home():
    welcome_msg = {
        "version": "1.0.0",
        "message": "Welcome to Flask API",
        "author": "Khairina Yasmine and I Nyoman Putra Maharddhika"
    }
    return jsonify(welcome_msg)

# Show sentiment analysis result Neural Network
@swag_from('docs/show_sentiment_result.yml', methods=['GET'])
@app.route('/show_sentiment_result', methods=['GET'])
def show_sentiment_result_api():
    db_connection = create_connection()
    sentiment_result = show_sentiment_result(db_connection)
    return jsonify(sentiment_result)

# Show sentiment analysis result LSTM
@swag_from('docs/show_sentiment_result_LSTM.yml', methods=['GET'])
@app.route('/show_sentiment_result_LSTM', methods=['GET'])
def show_sentiment_result_api_LSTM():
    db_connection = create_connection()
    sentiment_result = show_sentiment_result_LSTM(db_connection)
    return jsonify(sentiment_result)

# Neural Network sentiment analysis for text input
@swag_from('docs/neural_form.yml', methods=['POST'])
@app.route('/neural_network_form', methods=['POST'])
def neural_network_form():
    # Get text from input user
    raw_text = request.form["raw_text"]
    start = perf_counter()
    # Cleansing text
    clean_text = text_cleansing(raw_text)

    # Sentiment analysis
    sentiment_result = neural_sentiment(clean_text)
    end = perf_counter()
    time_elapse = end - start
    print(f"Processing time: {time_elapse} second")
    result_response = {"raw_text": raw_text, "clean_text": clean_text, "sentiment": sentiment_result, "processing_time": time_elapse}
    # Insert result to database
    db_connection = create_connection()
    insert_result_to_db(db_connection, raw_text, clean_text, sentiment_result)
    return jsonify(result_response)

# Neural Network sentiment analysis for file input
@swag_from('docs/neural_upload.yml', methods=['POST'])
@app.route('/neural_network_upload', methods=['POST'])
def neural_network_upload():
    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']
    # Read csv file upload, jika
    df_upload = pd.read_csv(uploaded_file, encoding="latin-1")

    start = perf_counter()
    # Read csv file to dataframe then cleansing
    result_df = neural_files(df_upload)
    end = perf_counter()
    time_elapse = end - start
    print(f"Processing time: {time_elapse} second")

    # Upload result to database
    db_connection = create_connection()
    insert_upload_result_to_db(db_connection, result_df)
    print("Upload result to database success!")
    print_result = result_df[["raw_text", "clean_text", "sentiment"]]
    result_response = print_result.T.to_dict()
    return jsonify(result_response)

# LSTM sentiment analysis for text input
@swag_from('docs/deep_learning_form.yml', methods=['POST'])
@app.route('/LSTM_form', methods=['POST'])
def deep_learning_form():
    # Get text from input user
    raw_text = request.form["raw_text"]
    # Cleansing text
    start = perf_counter()
    clean_text = text_cleansing(raw_text)
    sentiment_result = deep_learning(clean_text)
    end = perf_counter()
    time_elapse = end - start
    print(f"Processing time: {time_elapse} second")
    print(sentiment_result)
    result_response = {"raw_text": raw_text, "clean_text": clean_text, "sentiment": sentiment_result, "processing_time": time_elapse}
    # Insert result to database
    db_connection = create_connection()
    insert_result_to_db_LSTM(db_connection, raw_text, clean_text, sentiment_result)
    return jsonify(result_response)

# LSTM sentiment analysis for file input
@swag_from('docs/deep_learning_file.yml', methods=['POST'])
@app.route('/LSTM_file', methods=['POST'])
def deep_learning_file():
    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']
    # Read csv file upload, jika
    df_upload = pd.read_csv(uploaded_file, encoding="latin-1")
    # Read csv file to dataframe then cleansing
    start = perf_counter()
    df_sentiment = deep_learning_upload(df_upload)
    end = perf_counter()
    time_elapse = end - start
    print(f"Processing time: {time_elapse} second")
    # Upload result to database
    db_connection = create_connection()
    insert_upload_result_deep_learning_to_db(db_connection, df_sentiment)
    print("Upload result to database success!")
    print_result = df_sentiment[["raw_text", "clean_text", "sentiment"]]
    result_response = print_result.T.to_dict()
    return jsonify(result_response)

if __name__ == '__main__':
    app.run()
