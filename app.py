from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import random
import joblib
import os

app = Flask(__name__)

# Use raw strings or forward slashes for Windows paths
model_path = r'C:\Users\sivah\OneDrive\Desktop\consal\new\Inventory___management\server\ml_api\festival_demand_model.pkl'
model = joblib.load(model_path)

df = pd.read_csv("festival_data_modified.csv")

# Load encoders or fit and save them
le_festival = joblib.load("festival_encoder.pkl") if os.path.exists("festival_encoder.pkl") else None
le_product = joblib.load("product_encoder.pkl") if os.path.exists("product_encoder.pkl") else None

if le_festival is None or le_product is None:
    from sklearn.preprocessing import LabelEncoder
    le_festival = LabelEncoder()
    le_product = LabelEncoder()
    le_festival.fit(df['Festival'])
    le_product.fit(df['Product'])
    joblib.dump(le_festival, 'festival_encoder.pkl')
    joblib.dump(le_product, 'product_encoder.pkl')

# Generate synthetic festival dates
def get_festival_dates(year):
    def shift(base_day, variation=3):
        return base_day + random.randint(-variation, variation)

    return {
        "Deepavali": datetime(year, 10, shift(25)),
        "Vinayagar Chaturthi": datetime(year, 9, shift(10)),
        "Pongal": datetime(year, 1, 14),
        "Tamil New Year": datetime(year, 4, 14),
        "August Boost": datetime(year, 8, 15)
    }

# Prediction logic and pie chart generation
def predict_demand_for_year(year):
    fest_dates = get_festival_dates(year)
    gap_d_n = (fest_dates["Deepavali"] - fest_dates["Tamil New Year"]).days
    gap_p_n = (fest_dates["Pongal"] - fest_dates["Tamil New Year"]).days

    products = df['Product'].unique()
    festivals = df['Festival'].unique()

    predictions = []

    for fest in festivals:
        for product in products:
            row = {
                'Year': year,
                'Festival_Encoded': le_festival.transform([fest])[0],
                'Product_Encoded': le_product.transform([product])[0],
                'Month': fest_dates.get(fest, datetime(year, 6, 1)).month,
                'Is_August': 1 if fest == "August Boost" else 0,
                'Gap_Deepavali_NewYear': gap_d_n,
                'Gap_Pongal_NewYear': gap_p_n
            }
            X_input = pd.DataFrame([row])
            quantity_pred = model.predict(X_input)[0]
            predictions.append({'Festival': fest, 'Product': product, 'Predicted_Quantity': int(quantity_pred)})

    pred_df = pd.DataFrame(predictions)

    # Pie chart of total quantities per product
    product_totals = pred_df.groupby("Product")["Predicted_Quantity"].sum().reset_index()

    plt.figure(figsize=(6, 6))
    plt.pie(product_totals['Predicted_Quantity'], labels=product_totals['Product'], autopct='%1.1f%%')
    plt.title(f'Production Distribution for {year}')
    plt.savefig('static/pie_chart.png')
    plt.close()

    return predictions

# Main route
@app.route('/', methods=['GET', 'POST'])
def predict():
    predictions = None
    year = None
    if request.method == 'POST':
        year = int(request.form['year'])
        if year >= 2026:
            predictions = predict_demand_for_year(year)
    return render_template('predict.html', predictions=predictions, year=year)

if __name__ == '__main__':
    app.run(debug=True)
