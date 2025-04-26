from django.shortcuts import render
import joblib
import numpy as np
from .models import PredictionHistory

def home(request):
    return render(request, 'prediction/home.html')

def about(request):
    return render(request, 'prediction/about.html')

def contact(request):
    return render(request, 'prediction/contact.html')

def login_page(request):
    return render(request, 'prediction/login.html')



# Load model, scaler, encoders
model = joblib.load('./ML_part/survival_model.pkl')
scaler = joblib.load('./ML_part/scaler.pkl')
label_encoders = joblib.load('./ML_part/label_encoders.pkl')


def predict_survival(request):
    if request.method == 'POST':
        try:
            # expecting clinical + gene + protein inputs
            age = int(request.POST['age'])
            treatment_type = request.POST['treatment_type']
            tumor_stage = request.POST['tumor_stage']

            # Encode categorical fields
            treatment_encoded = label_encoders['treatment_type'].transform([treatment_type])[0]
            tumor_stage_encoded = label_encoders['tumor_stage'].transform([tumor_stage])[0]

            # For simplicity: dummy gene/protein values here
            gene_values = [float(request.POST.get(f'gene_{i}', 0.5)) for i in range(5000)]
            protein_values = [float(request.POST.get(f'protein_{i}', 0.5)) for i in range(300)]

            # Combine all inputs
            input_features = gene_values + protein_values + [age, treatment_encoded, tumor_stage_encoded]
            input_scaled = scaler.transform([input_features])

            # Predict survival
            prediction = model.predict(input_scaled)[0]
            survival_chance = "Survived" if prediction == 1 else "Did Not Survive"

            return render(request, 'prediction/result.html', {'result': survival_chance})

        except Exception as e:
            return render(request, 'prediction/result.html', {'result': f'Error: {e}'})

    return render(request, 'prediction/form.html')
