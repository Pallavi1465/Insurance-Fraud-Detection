from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv('data/processed/cleaned_insurance_data.csv')
data.drop(columns=["_c39"], inplace=True, errors='ignore')

X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


categorical_features = [
    'policy_state','insured_sex','insured_education_level','insured_occupation',
    'insured_hobbies','insured_relationship','incident_type','collision_type',
    'incident_severity','authorities_contacted','incident_state','incident_city',
    'property_damage','police_report_available','auto_make','auto_model','policy_csl_limit'
    ]
numerical_features = [
    'months_as_customer','age','policy_deductable','policy_annual_premium','umbrella_limit',
    'capital-gains','capital-loss','incident_hour_of_the_day','number_of_vehicles_involved',
    'bodily_injuries','witnesses','total_claim_amount','injury_claim','property_claim',
    'vehicle_claim','auto_year','claim_to_premium_ratio'
    ]
#processor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough'
)
#pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier( criterion='gini', min_samples_split=2,
                                      class_weight='balanced', random_state=12))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test, y_pred))
#Save the model
import pickle
with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
