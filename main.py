from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
model = pickle.load(open('model/best_model.pkl', 'rb'))

# All features used during training
FEATURE_COLUMNS = [
'months_as_customer','age','policy_state','policy_deductable',
'policy_annual_premium','umbrella_limit','insured_sex',
'insured_education_level','insured_occupation','insured_hobbies',
'insured_relationship','capital-gains','capital-loss','incident_type',
'collision_type','incident_severity','authorities_contacted',
'incident_state','incident_city','incident_hour_of_the_day',
'number_of_vehicles_involved','property_damage','bodily_injuries',
'witnesses','police_report_available','total_claim_amount',
'injury_claim','property_claim','vehicle_claim','auto_make',
'auto_model','auto_year','claim_to_premium_ratio','policy_csl_limit'
]

# Default values for advanced settings
DEFAULT_VALUES = {
"policy_state":"OH",
"policy_deductable":500,
"umbrella_limit":0,
"insured_sex":"MALE",
"insured_education_level":"Bachelors",
"insured_occupation":"other",
"insured_hobbies":"reading",
"insured_relationship":"own-child",
"capital-gains":0,
"capital-loss":0,
"incident_type":"Single Vehicle Collision",
"collision_type":"Front Collision",
"authorities_contacted":"Police",
"incident_state":"NY",
"incident_city":"New York",
"incident_hour_of_the_day":12,
"property_damage":"NO",
"police_report_available":"YES",
"injury_claim":0,
"property_claim":0,
"vehicle_claim":0,
"auto_make":"Toyota",
"auto_model":"Corolla",
"claim_to_premium_ratio":1,
"policy_csl_limit":300000
}


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def prediction():

    input_data = {}

    for col in FEATURE_COLUMNS:

        value = request.form.get(col)

        if value is None or value == "":
            value = DEFAULT_VALUES.get(col)

        input_data[col] = value

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]
    probability = round(probability * 100, 2)

    return render_template(
        "results.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)