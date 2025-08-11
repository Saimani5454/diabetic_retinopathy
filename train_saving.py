import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load dataset
df = pd.read_csv('pronostico_dataset (1).csv', sep=';')

# Fix datatypes
for col in ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol']:
    df[col] = df[col].astype(int)

# Drop unnecessary column
df.drop('ID', axis=1, inplace=True)

# Scale features
numerical_cols = ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Encode target
label_encoder = LabelEncoder()
df['prognosis'] = label_encoder.fit_transform(df['prognosis'])

# Split data
X = df.drop('prognosis', axis=1)
y = df['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# Save model, scaler, and label encoder
with open('svm_model.sav', 'wb') as f:
    pickle.dump(mo
