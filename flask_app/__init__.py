import re
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open("C:/Users/Songhee/bootcamp/Section3/Section3_project/flask_app/models/model_1.pkl",'rb'))

    report_date = request.form['report_date']
    gu = request.form['gu']
    dong = request.form['dong']
    detail_address = request.form['detail_address']
    category = request.form['category']

    report_date_b = pd.to_datetime(report_date)
    report_date_b = int(round(report_date_b.timestamp()))

    test_data = pd.DataFrame(data = [[report_date_b, gu, dong, category]], columns=['신고일','구정보','동정보', '유형'])


    pred = model.predict(test_data)

    context = {
        'report_date':report_date,
        'report_data_b':report_date_b,
        'gu':gu,
        'dong':dong,
        'detail_address':detail_address,
        'category':category,
        'pred':pred
    }

    return render_template('predict.html', data=context)

if __name__ == '__main__':
    # model = pickle.load(open("C:/Users/Songhee/bootcamp/Section3/Section3_project/flask_app/models/model_1.pkl",'rb'))
    app.run(debug=True)
    