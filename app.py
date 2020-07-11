# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))
model4 = pickle.load(open('model4.pkl', 'rb'))
a = model.predict([[2.6,1360,1046,116,1056,113,1692,1268,13.6,48.9,0.7578]])
print(a)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictLR',methods=['post'])
def predictLR():
    return render_template('linear.html')

@app.route('/predictLRR',methods=['POST'])
def predictLRR():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.values()
    final_features = [list(map(float,int_features))]
    prediction = model.predict(final_features)

    output = round(prediction[0][0], 2)

    return render_template('linear.html', prediction_text='Amount Of C6H6 is % {}'.format(output))
@app.route('/predictDR',methods=['post'])
def predictDR():
    return render_template('decision.html')
@app.route('/predictDRR',methods=['POST'])
def predictDRR():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.values()
    final_features = [list(map(float,int_features))]
    prediction = model2.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('decision.html', prediction_text='Amount Of C6H6 is % {}'.format(output))
@app.route('/predictRR',methods=['post'])
def predictRR():
    return render_template('random.html')
@app.route('/predictRRR',methods=['POST'])
def predictRRR():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.values()
    final_features = [list(map(float,int_features))]
    prediction = model3.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('random.html', prediction_text='Amount Of C6H6 is % {}'.format(output))

@app.route('/predictGR',methods=['post'])
def predictGR():
    return render_template('gregressor.html')
@app.route('/predictGRR',methods=['POST'])
def predictGRR():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.values()
    final_features = [list(map(float,int_features))]
    prediction = model4.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('gregressor.html', prediction_text='Amount Of C6H6 is % {}'.format(output))

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)