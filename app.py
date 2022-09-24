from flask import Flask,render_template,request
import numpy as np
import pickle

model = pickle.load(open('artifact/linear.pkl','rb'))
scale = pickle.load(open('artifact/scale.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form
    input_data=np.zeros(13)
    input_data[0]=data['A']
    input_data[1]=data['B']
    input_data[2]=data['C']
    input_data[3]=data['D']
    input_data[4]=data['E']
    input_data[5]=data['F']
    input_data[6]=data['G']
    input_data[7]=data['H']
    input_data[8]=data['I']
    input_data[9]=data['J']
    input_data[10]=data['K']
    input_data[11]=data['L']
    input_data[12]=data['M']

    s= scale.transform([input_data])
    p= model.predict(s)
    print(p)

    return render_template('input.html',price=p)

if __name__=='__main__':
    app.run(debug=True)