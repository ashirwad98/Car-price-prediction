from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
load = pickle.load(open('carprc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        car_ID = request.form['car_ID']
        fueltype = request.form['fueltype']
        aspiration = request.form['aspiration']
        doornumber = request.form['doornumber']
        drivewheel = request.form['drivewheel']
        cylindernumber = request.form['cylindernumber']
        horsepower = request.form['horsepower']
        citympg = request.form['citympg']
        highwaympg =request.form['highwaympg']
        

        feature_list = [car_ID, fueltype, aspiration, doornumber, drivewheel, cylindernumber, horsepower,citympg,highwaympg]
        single_pred = np.array(feature_list).reshape(1, -1)

        prediction = load.predict(single_pred)
        output = prediction[0]

        return render_template('index.html', prediction=output)

    except ValueError:
        return render_template('index.html', prediction="Error: Please enter valid numeric values for all fields.")

if __name__ == "__main__":
    app.run(debug=True)

