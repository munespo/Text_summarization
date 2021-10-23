from  flask import Flask, render_template, request
import joblib
import re
from model import TextSummarize
import math

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def predict():

    #text = request.form.get('text')
    textfile = request.files['textfile']
    text_path = "./text" + textfile.filename
    textfile.save(text_path)
    text = open(text_path,"r")
    article_text = text.read()
    summary = preprocessDataAndSummarize(article_text)
    

    return render_template('index.html', input = article_text, prediction = summary)


def preprocessDataAndSummarize(text):

    #preprocess by removing extra spaces and /\ in the middle of sentences
    article_text = re.sub(r'\[[0-9]*\]', ' ', text)
    article_text = re.sub(r'\s+', ' ', article_text)
    article_text=article_text.replace("\"", "")

    #open file
    # file_model = open('./saved_models/text_summarizer.pkl', "rb")

    #load file
    textsummary = TextSummarize(article_text, threshold=0.01,ratio=0.5)
    summary = textsummary.summary
    return summary







if __name__ == '__main__':
    app.run(debug=True)