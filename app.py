from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

labels = pd.read_csv('static/encoders/labels.csv', names=['crop','state','season'])

@app.route("/", methods=["GET", "POST"])                 
def home():
    algorithms = {'Decision Tree Regressor':90.14,'Random Forest Regressor':94.62}
    states = labels['state'].dropna()
    crops = labels['crop'].dropna()
    seasons = labels['season'].dropna()
    production,accuracy= [0], 0
    if request.method == "POST":
        state = request.form["state"]
        year = request.form["year"]
        season = request.form["season"]
        crop = request.form["crop"]
        area = request.form["area"]
        if request.form['algorithm']=='Decision Tree Regressor':
            production = predict([state, year, season, crop, area],1)
            accuracy = algorithms['Decision Tree Regressor']
        elif request.form['algorithm']=='Random Forest Regressor':
            production = predict([state, year, season, crop, area],2)
            accuracy = algorithms['Random Forest Regressor']

    return render_template("index.html", algorithms=algorithms.keys(), 
                           states=states, crops=crops, seasons=seasons, production=production[0], accuracy=accuracy)     

def predict(input,ch):
    crop_enc = pickle.load(open('static/encoders/crop.enc','rb'))
    season_enc = pickle.load(open('static/encoders/season.enc','rb'))
    state_enc = pickle.load(open('static/encoders/state.enc','rb'))
    state = state_enc.transform([input[0]])
    season = season_enc.transform([input[2]])
    crop = crop_enc.transform([input[3]])
    if ch==1:
        RandomForest = pickle.load(open('static/model/DTR.h5','rb'))
        production = RandomForest.predict([[state, input[1], season, crop, input[4]]])
        return production
    if ch==2:
        RandomForest = pickle.load(open('static/model/RFR.h5','rb'))
        production = RandomForest.predict([[state, input[1], season, crop, input[4]]])
        return production
if __name__ == "__main__":        # on running python app.py
    app.run(debug=True)                     # run the flask app
