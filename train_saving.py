import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import pickle
import time

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Speed-up for testing: sample smaller training set
DEBUG_MODE = True  # Set to False for full training
if DEBUG_MODE:
    X_train = X_train.sample(200, random_state=42)
    y_train = y_train.loc[X_train.index]

# Train SVM
start_time = time.time()

# Use probability=False for speed, wrap later if needed
svm_model = SVC(kernel='rbf', probability=False, random_state=42)
svm_model.fit(X_train, y_train)

# If probabilities are required, calibrate the model
calibrated_model = CalibratedClassifierCV(svm_model, cv=3)
calibrated_model.fit(X_train, y_train)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Save model, scaler, and label encoder
with open('svm_model.sav', 'wb') as f:
    pickle.dump(calibrated_model, f)

with open('scaler.sav', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.sav', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model, scaler, and label encoder saved successfully!")
