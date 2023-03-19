from flask import Flask, render_template, jsonify, request, redirect
from flask_restful import Api, Resource, reqparse
import pandas as pd
import numpy as np
import json5
import pickle
import sklearn
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')


# Define how the api will respond to the post requests
class SpamEmailClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        # variant 1
        json = json5.loads(args['data'])
        lst = [[json['subject'], json['message']]]
        df_res = pd.DataFrame(lst, columns=['subject', 'message'])
        res = prepare_data_transform(df_res)
        res_dic = {"prediction": res}
        return jsonify(res_dic)


api.add_resource(SpamEmailClassifier, '/api/v1/predict-spam-email')


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        subject = request.form['subject']
        message = request.form['message']
        lst = [[subject, message]]
        df_res = pd.DataFrame(lst, columns=['subject', 'message'])
        res = prepare_data_transform(df_res)
    else:
        return redirect('/')
    # Make predict
    # Return result in html
    return render_template('predict.html', prediction=res)


if __name__ == '__main__':
    app.run()


def prepare_data_transform(df_res):
    df_res['sub_mssg'] = df_res['subject'] + df_res['message']
    df_res['length'] = df_res['sub_mssg'].apply(len)
    df_res.drop('subject', axis=1, inplace=True)
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'\d+(\.\d+)?', 'numbers')

    # CONVRTING EVERYTHING TO LOWERCASE
    df_res['sub_mssg'] = df_res['sub_mssg'].str.lower()
    # REPLACING NEXT LINES BY 'WHITE SPACE'
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'\n', " ")
    # REPLACING EMAIL IDs BY 'MAILID'
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'MailID')
    # REPLACING URLs  BY 'Links'
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'Links')
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'£|\$', 'Money')
    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'\s+', ' ')

    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'^\s+|\s+?$', '')
    # REPLACING CONTACT NUMBERS
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'contact number')
    # REPLACING SPECIAL CHARACTERS  BY WHITE SPACE
    df_res['sub_mssg'] = df_res['sub_mssg'].str.replace(r"[^a-zA-Z0-9]+", " ")

    # CONVRTING EVERYTHING TO LOWERCASE
    df_res['message'] = df_res['message'].str.lower()
    # REPLACING NEXT LINES BY 'WHITE SPACE'
    df_res['message'] = df_res['message'].str.replace(r'\n', " ")
    # REPLACING EMAIL IDs BY 'MAILID'
    df_res['message'] = df_res['message'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'MailID')
    # REPLACING URLs  BY 'Links'
    df_res['message'] = df_res['message'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'Links')
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df_res['message'] = df_res['message'].str.replace(r'£|\$', 'Money')
    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df_res['message'] = df_res['message'].str.replace(r'\s+', ' ')

    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df_res['message'] = df_res['message'].str.replace(r'^\s+|\s+?$', '')
    # REPLACING CONTACT NUMBERS
    df_res['message'] = df_res['message'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'contact number')
    # REPLACING SPECIAL CHARACTERS  BY WHITE SPACE
    df_res['message'] = df_res['message'].str.replace(r"[^a-zA-Z0-9]+", " ")

    stop = stopwords.words('english')
    df_res['Cleaned_Text'] = df_res['sub_mssg'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    df_res.drop('message', axis=1, inplace=True)
    df_res.drop('sub_mssg', axis=1, inplace=True)
    df_res['lgth_clean'] = df_res['Cleaned_Text'].apply(len)
    df = df_res.Cleaned_Text
    # Load model
    with open('model_v2.pickle', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(df)
    return prediction.tolist()[0]
