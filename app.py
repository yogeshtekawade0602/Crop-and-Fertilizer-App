from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# =========================
# Load Models
# =========================
crop_model = pickle.load(open('crop_model.pkl', 'rb'))
fertilizer_model = pickle.load(open('fertilizer_model.pkl', 'rb'))

# =========================
# Columns
# =========================
columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature',
           'District_Name_Kolhapur', 'District_Name_Pune', 'District_Name_Sangli',
           'District_Name_Satara', 'District_Name_Solapur',
           'Soil_color_Black', 'Soil_color_Dark Brown', 'Soil_color_Light Brown',
           'Soil_color_Medium Brown', 'Soil_color_Red',
           'Soil_color_Reddish Brown',
           'Season_Kharip', 'Season_Rabi',
           'Season_Summer', 'Season_Whole Year']

fert_columns = ['Crop_Predict'] + columns

crop_dict = {
    'Sugarcane':1,'Wheat':2,'Cotton':3,'Jowar':4,'Maize':5,'Rice':6,
    'Groundnut':7,'Tur':8,'Ginger':9,'Grapes':10,'Urad':11,
    'Moong':12,'Gram':13,'Turmeric':14,'Soybean':15,'Masoor':16
}

reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# =========================
# Home
# =========================
@app.route('/')
def home():
    return render_template('home.html')

# =========================
# Crop Page
# =========================
@app.route('/crop', methods=['GET', 'POST'])
def crop():

    crop_result = None

    if request.method == 'POST':

        input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        input_df.loc[0, 'Nitrogen'] = float(request.form['nitrogen'])
        input_df.loc[0, 'Phosphorus'] = float(request.form['phosphorus'])
        input_df.loc[0, 'Potassium'] = float(request.form['potassium'])
        input_df.loc[0, 'pH'] = float(request.form['ph'])
        input_df.loc[0, 'Rainfall'] = float(request.form['rainfall'])
        input_df.loc[0, 'Temperature'] = float(request.form['temperature'])

        input_df.loc[0, f"District_Name_{request.form['district']}"] = 1
        input_df.loc[0, f"Soil_color_{request.form['soil']}"] = 1
        input_df.loc[0, f"Season_{request.form['season']}"] = 1

        prediction = crop_model.predict(input_df)
        crop_result = reverse_crop_dict.get(prediction[0], "Prediction Error")

    return render_template('crop.html', crop_result=crop_result)

# =========================
# Fertilizer Page
# =========================
@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():

    crops = list(crop_dict.keys())   # 👈 ADD THIS LINE
    fertilizer_result = None

    if request.method == 'POST':

        input_df = pd.DataFrame(np.zeros((1, len(fert_columns))), columns=fert_columns)

        input_df.loc[0, 'Crop_Predict'] = crop_dict[request.form['crop']]
        input_df.loc[0, 'Nitrogen'] = float(request.form['nitrogen'])
        input_df.loc[0, 'Phosphorus'] = float(request.form['phosphorus'])
        input_df.loc[0, 'Potassium'] = float(request.form['potassium'])
        input_df.loc[0, 'pH'] = float(request.form['ph'])
        input_df.loc[0, 'Rainfall'] = float(request.form['rainfall'])
        input_df.loc[0, 'Temperature'] = float(request.form['temperature'])

        input_df.loc[0, f"District_Name_{request.form['district']}"] = 1
        input_df.loc[0, f"Soil_color_{request.form['soil']}"] = 1
        input_df.loc[0, f"Season_{request.form['season']}"] = 1

        prediction = fertilizer_model.predict(input_df)
        fertilizer_result = prediction[0]

    return render_template(
        'fertilizer.html',
        fertilizer_result=fertilizer_result,
        crops=crops     # 👈 PASS TO HTML
    )

if __name__ == '__main__':
    app.run(debug=True)

# For Vercel
app = app