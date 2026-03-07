import pandas as pd
import numpy as np

# Load
df = pd.read_csv('data/insurance_claims.csv')

# 1. Identify '?' as actual missing values
df.replace('?', np.nan, inplace=True)

# 2. Impute (Fill) Missing Values
# For Categorical: use 'mode' (most frequent)
# For Numerical: use 'mean' or 'median'
df['collision_type'].fillna(df['collision_type'].mode()[0], inplace=True)
df['property_damage'].fillna('NO', inplace=True) # Usually '?' means no damage reported
df['police_report_available'].fillna('NO', inplace=True)

print("Missing values handled.")
# Drop columns that don't contribute to fraud patterns
cols_to_drop = [
    'policy_number',      # Unique ID
    'policy_bind_date',   # Specific date (high cardinality)
    'incident_date',      # Specific date
    'incident_location',  # Thousands of unique addresses
    'insured_zip'         # Too many unique values
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
# Example: Ratio of claim amount to annual premium
df['claim_to_premium_ratio'] = df['total_claim_amount'] / (df['policy_annual_premium'] + 1)

# Example: Converting policy_csl (e.g. '250/500') into a numerical limit
df['policy_csl_limit'] = df['policy_csl'].str.split('/').str[0].astype(int)
df.drop('policy_csl', axis=1, inplace=True)

print("Feature engineering complete.")
from sklearn.preprocessing import LabelEncoder

# Apply Label Encoding to all 'object' (text) columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print(f"Encoded {len(categorical_cols)} categorical columns.")
# Create folder if it doesn't exist
import os
os.makedirs('data/processed', exist_ok=True)

# Save
df.to_csv('data/processed/cleaned_insurance_data.csv', index=False)
print("Preprocessed data saved to data/processed/")