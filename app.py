import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder=".", template_folder=".")

# Global variable to hold the trained model artifacts
MODEL_ARTIFACTS = None
MODEL_PATH = "cardioguard_model.pkl"

def load_model():
    global MODEL_ARTIFACTS
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                MODEL_ARTIFACTS = pickle.load(f)
            print("Successfully loaded CardioGuard AI model artifacts.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
    else:
        print(f"Model file '{MODEL_PATH}' not found. Please run 'train_model.py' first.")

# Load the model on startup
load_model()

@app.route('/')
def index():
    # Serve the main HTML dashboard
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    global MODEL_ARTIFACTS
    if MODEL_ARTIFACTS is None:
        # Try to load again
        load_model()
        if MODEL_ARTIFACTS is None:
            return jsonify({
                "status": "error",
                "message": "Model not trained or loaded. Please run train_model.py first."
            }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No input data provided"}), 400

        # Required features list
        required_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Validate inputs
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                "status": "error",
                "message": f"Missing parameters: {', '.join(missing_features)}"
            }), 400

        # Convert input into a single-row DataFrame
        patient_data = {feat: [float(data[feat])] for feat in required_features}
        df_patient = pd.DataFrame(patient_data)

        # Unpack model artifacts
        model = MODEL_ARTIFACTS['model']
        scaler = MODEL_ARTIFACTS['scaler']
        continuous_cols = MODEL_ARTIFACTS['continuous_cols']
        feature_importances = MODEL_ARTIFACTS['feature_importances']

        # Scale continuous features
        df_patient_scaled = df_patient.copy()
        df_patient_scaled[continuous_cols] = scaler.transform(df_patient[continuous_cols])

        # Get probability prediction
        # class 1: High Risk (Heart Disease)
        risk_probability = float(model.predict_proba(df_patient_scaled)[0, 1])
        prediction = int(model.predict(df_patient_scaled)[0])

        # Determine risk category
        if risk_probability < 0.35:
            risk_category = "Low"
            risk_color = "green"
        elif risk_probability < 0.70:
            risk_category = "Moderate"
            risk_color = "yellow"
        else:
            risk_category = "High"
            risk_color = "red"

        # Calculate clinical feature contributions (XAI)
        # We can calculate custom contribution scores by comparing this patient's values
        # against a clinical baseline, scaled by feature importance.
        # Baselines:
        baselines = {
            'age': 50,
            'sex': 0,
            'cp': 1, # non-anginal / atypical
            'trestbps': 120,
            'chol': 200,
            'fbs': 0,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 0.0,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }

        contributions = {}
        for feat in required_features:
            val = float(data[feat])
            base = baselines[feat]
            importance = feature_importances[feat]

            # Custom contribution logic based on feature behavior
            if feat == 'age':
                diff = (val - base) / 30.0
            elif feat == 'trestbps':
                diff = (val - base) / 40.0
            elif feat == 'chol':
                diff = (val - base) / 100.0
            elif feat == 'thalach':
                # Higher heart rate is actually protective, so negative difference is risky
                diff = (base - val) / 50.0
            elif feat == 'oldpeak':
                diff = (val - base) / 2.0
            elif feat in ['exang', 'fbs', 'sex']:
                diff = 1.0 if val > 0 else -0.5
            elif feat == 'ca':
                diff = val / 2.0 if val > 0 else -0.5
            elif feat == 'cp':
                diff = 1.0 if val == 3 else (0.5 if val == 0 else -0.5)
            elif feat == 'thal':
                diff = 1.0 if val == 3 else (0.5 if val == 2 else -0.5)
            else:
                diff = (val - base)
            
            # Weighted contribution
            contrib = diff * importance * 100
            contributions[feat] = round(contrib, 2)

        # Generate personalized, actionable clinical recommendations
        recommendations = []
        
        # 1. Blood Pressure Recommendation
        bp = float(data['trestbps'])
        if bp >= 140:
            recommendations.append("Hypertension detected (BP >= 140 mmHg). Recommend dietary sodium restriction (< 1.5g/day), daily cardiovascular exercise, and a consultation for blood pressure management.")
        elif bp >= 130:
            recommendations.append("Elevated Blood Pressure (130-139 mmHg) detected. Initiate lifestyle modifications: increase potassium-rich foods, reduce stress, and monitor BP bi-weekly.")

        # 2. Cholesterol Recommendation
        cholesterol = float(data['chol'])
        if cholesterol >= 240:
            recommendations.append("Hypercholesterolemia (Cholesterol >= 240 mg/dL). Strongly advise a low-saturated-fat Mediterranean diet, lipid panel re-evaluation in 8 weeks, and discussion of statin therapy with a physician.")
        elif cholesterol >= 200:
            recommendations.append("Borderline High Cholesterol (200-239 mg/dL). Increase dietary soluble fiber (oats, legumes) and omega-3 fatty acids.")

        # 3. Fasting Blood Sugar
        if float(data['fbs']) == 1:
            recommendations.append("Elevated Fasting Blood Sugar (>120 mg/dL). Consistent with potential diabetes/impaired glucose tolerance. Recommend HbA1c screening and carbohydrate control.")

        # 4. Chest Pain & Exercise Angina
        if float(data['exang']) == 1 or float(data['cp']) == 3:
            recommendations.append("Signs of inducible ischemia or chest pain symptoms. Strongly advise a clinical Cardiology consultation and potentially an Exercise Tolerance Test (ETT) or Stress Echocardiogram.")

        # 5. Heart Rate
        if float(data['thalach']) < 120 and float(data['age']) < 65:
            recommendations.append("Low maximum heart rate achieved. Consider checking for beta-blocker usage or assessing chronotropic incompetence.")

        # 6. Blocked Vessels & ST-Depression
        if float(data['ca']) >= 1 or float(data['oldpeak']) >= 1.5:
            recommendations.append("Significant diagnostic indicators of Coronary Artery Disease present (elevated oldpeak / positive fluoroscopy). A Coronary Angiography should be considered for definitive structural evaluation.")

        # Default encouraging recommendation if low risk and no red flags
        if len(recommendations) == 0:
            recommendations.append("Patient parameters are within normal physiological thresholds. Encourage continued adherence to AHA/ACC guidelines: 150 minutes of moderate-intensity exercise weekly and a balanced whole-foods diet.")

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "risk_score": round(risk_probability, 4),
            "risk_score_percentage": round(risk_probability * 100, 1),
            "risk_category": risk_category,
            "risk_color": risk_color,
            "recommendations": recommendations,
            "feature_contributions": contributions
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"An error occurred during prediction: {str(e)}"
        }), 500

if __name__ == "__main__":
    print("Starting CardioGuard AI Flask Server...")
    # Serve on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
