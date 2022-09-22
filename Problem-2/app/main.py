# import objects from the Flask model
from flask import Flask, jsonify, render_template, request, make_response
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
# creating flask app
app = Flask(__name__)

checkpoint = "./tokenizer"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained('./model', num_labels=5)

# inference fonction
def get_prediction(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    results = model(tokens)
    predict_sentiment = int(torch.argmax(results.logits))+1
    return predict_sentiment

# get method to load the home page
@app.route('/')
def get():
    return render_template('home.html')


# post method for prediction
@app.route('/', methods=['POST'])
def predict():
    message = request.form['message']
    results = get_prediction(message)
    my_prediction = f'The rating of review is {results} '
 
    return render_template('result.html', text = f'{message}', prediction = my_prediction)

# post method for getting response in form of json
@app.route('/predict', methods=['POST'])
def predict1():
    message = request.json['message']
    results = get_prediction(message)
    my_prediction = f'The rating of review is {results} '
 
    return jsonify({'Star Rating': my_prediction})



