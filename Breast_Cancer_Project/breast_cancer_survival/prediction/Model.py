import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from django.db import models

class PredictionHistory(models.Model):
    age = models.IntegerField()
    treatment_type = models.CharField(max_length=50)
    tumor_stage = models.CharField(max_length=50)
    prediction_result = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)

# ------------------ Load Datasets ------------------
transcriptomics = pd.read_csv("../ML_part/transcriptomics_synthetic.csv")
proteomics = pd.read_csv("../ML_part/proteomics_synthetic.csv")
clinical = pd.read_csv("../ML_part/clinical_synthetic.csv")

# ------------------ Merge Datasets ------------------
merged_df = transcriptomics.merge(proteomics, on='patient_id') \
                           .merge(clinical, on='patient_id')

print(f"[INFO] Data merged successfully. Shape: {merged_df.shape}")

# ------------------ Encode Categorical Columns ------------------
label_encoders = {}

# Loop through all object (text) columns except 'patient_id' and target
for column in merged_df.columns:
    if merged_df[column].dtype == 'object' and column not in ['patient_id']:
        le = LabelEncoder()
        merged_df[column] = le.fit_transform(merged_df[column])
        label_encoders[column] = le

print("[INFO] Categorical columns encoded.")

# ------------------ Split Features and Target ------------------
y = merged_df['survival_status']  # Target variable
X = merged_df.drop(['patient_id', 'survival_status'], axis=1)  # Features

# ------------------ Scale Features ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------ Train Random Forest Model ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------ Evaluate Model ------------------
y_pred = model.predict(X_test)
print("\n[MODEL REPORT]")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# ------------------ Save Model, Scaler, and Encoders ------------------
joblib.dump(model, '../ML_part/survival_model.pkl')
joblib.dump(scaler, '../ML_part/scaler.pkl')
joblib.dump(label_encoders, '../ML_part/label_encoders.pkl')  # Save encoders too!

print("\n[SAVED] Model, scaler, and encoders saved for Django integration.")
