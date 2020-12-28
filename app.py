from flask import Flask, url_for, request, jsonify, render_template, redirect
import pickle
import os
import numpy as np

# Create application
app = Flask(__name__)

#load saved model
def load_model():
  return pickle.load(open('naive_bayes.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

#predict result
#fe1,fe2,fe3,fe4,fe5,fe6,fe7,fe8,fe9,fe10,fe11
@app.route('/predict', methods=["POST", "GET"])
def predict():
  labels = ['Rejected', 'Approved']
  features = []
  if request.method == "POST":
    fe1 = request.form["ApplicantIncome"]
    features.append(float(fe1))
    fe2 = request.form['CoapplicantIncome']
    features.append(float(fe2))
    fe3 = request.form['LoanAmount']
    features.append(float(fe3))
    fe4 = request.form['Loan_Amount_Term']
    features.append(float(fe4))
    fe5 = request.form['Credit_History']
    features.append(int(fe5))
    
    fe6 = int(request.form['Gender'])
    if fe6 == 0: fe6 = [1,0,0]
    elif fe6 == 1: fe6 = [0,1,0]
    else: fe6 = [0,0,1]
    features.extend(fe6)

    fe7 = int(request.form['Married'])
    if fe7 == 0: fe7 = [1,0,0]
    elif fe7 == 1: fe7 = [0,1,0]
    else: fe7 = [0,0,1]
    features.extend(fe7)
    
    fe8 = int(request.form['Dependents'])
    if fe8 == 0: fe8 = [1,0,0,0,0]
    elif fe8 == 1: fe8 = [0,1,0,0,0]
    elif fe8 == 2: fe8 = [0,0,1,0,0]
    elif fe8 == 3: fe8 = [0,0,0,1,0]
    else: fe8 = [0,0,0,0,1]
    features.extend(fe8)

    fe9 = int(request.form['Education'])
    if fe9 == 0: fe9 = [1,0]
    else: fe9 = [0,1]
    features.extend(fe9)

    fe10 = int(request.form['Self_Employed'])
    if fe10 == 0: fe10 = [1,0,0]
    elif fe10 == 1: fe10 = [0,1,0]
    else: fe10 = [0,0,1]
    features.extend(fe10)

    fe11 = int(request.form['Property_Area'])
    if fe11 == 0: fe11 = [1,0,0]
    elif fe11 == 1: fe11 = [0,1,0]
    else: fe11 = [0,0,1]
    features.extend(fe11)
  
  drop_feature = [1,3,5,9,15]
  features = np.delete(features, drop_feature)

  model = load_model()
  prediction = model.predict([features])

  result = labels[prediction[0]]
  return render_template('index.html', output= 'Loan prediction: {}'.format(result))

if __name__ == "__main__":
  # port = int(os.environ.get('PORT', 5000))
  # app.run(port=port, debug=True,use_reloader=False)
  app.run(debug=True)