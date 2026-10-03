
# CardioGuard AI: Clinical Decision Support & Risk Stratification Dashboard

**CardioGuard AI** is a production-ready, full-stack Artificial Intelligence Engineer mini-project designed for the healthcare domain. It showcases how to combine **Machine Learning (Scikit-Learn)**, **Explainable AI (XAI)**, and **Clinical Decision Support (CDSS)** into an interactive, clinician-facing web application.

This project uses clinical parameters (age, blood pressure, cholesterol, resting ECG, fluoroscopy vessels, etc.) to predict the likelihood of cardiovascular disease and provides **interpretable feature contributions** and **personalized clinical recommendations** based on medical guidelines.

---

## 🌟 Key Features

1. **Synthetic Clinical Data Generator (`generate_data.py`)**:
   - Synthesizes a dataset of 1,200 patient profiles with realistic biological correlations (e.g., age-correlated heart rate decline, chest pain severity, ST depression pathological indicators).
   - Mimics the schema of the famous *UCI Cleveland Heart Disease dataset*.

2. **Machine Learning Pipeline (`train_model.py`)**:
   - Implements data split, feature scaling using `StandardScaler` for continuous features, and trains a **Random Forest Classifier**.
   - Evaluates the model using rich diagnostics: **Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC**.
   - Saves trained model artifacts (`cardioguard_model.pkl`) using Python's standard serialization (`pickle`).

3. **RESTful Flask API Backend (`app.py`)**:
   - Implements a lightweight web server exposing a `/api/predict` endpoint.
   - Performs standard scaling and infers risk percentages on-the-fly.
   - Calculates **local feature contribution scores (XAI)** and maps results to structured **lifestyle and medical guidelines**.

4. **Stunning Interactive Dashboard UI (`index.html`)**:
   - A polished, modern, clinical design that runs **100% offline** and is fully sandboxed-compatible (using inline styles and zero external script downloads).
   - **Dual-Mode execution**: Detects whether the local Flask server is running; if not, it automatically runs an embedded high-fidelity JavaScript clinical engine matching the Python model's weights.
   - **One-Click Patient Case Studies**: Load 4 diverse profiles (from low-risk active athletes to high-risk diabetic or blocked-vessel elderly patients) to see immediate changes in risk scores, XAI graphs, and medical advice.
   - **Dynamic HTML5 Canvas Gauge**: Animates a circular medical needle dial reflecting the risk percentage and color-grading (Green, Yellow, Red).
   - **SHAP-inspired Local Explanation Charts**: Renders custom horizontal bar charts illustrating exactly which factors increased (red) or decreased (green) the patient's individual risk.

---

## 📂 Project Architecture & Directories

```text
cardioguard_ai/
│
├── generate_data.py       # Synthesizes realistic cardiovascular clinical dataset (CSV)
├── train_model.py         # Loads data, pre-processes, trains Random Forest, saves pickle
├── app.py                 # Flask web backend exposing REST API and serving UI
├── index.html             # State-of-the-art interactive front-end web app
├── requirements.txt       # Python package dependencies
└── heart_disease_data.csv # Generated dataset (created upon first script run)
```

---

## 🔬 Scientific & Clinical Formulation

### 1. The Risk Scoring Engine (Logit Framework)
To ensure synthetic data mimics biological reality, the simulator is based on a logit equation representing log-odds of coronary artery disease:

$$z = \beta_0 + \beta_{age}(\text{age}) + \beta_{sex}(\text{sex}) + \beta_{cp}(\text{chest pain}) + \beta_{bp}(\text{blood pressure}) + \dots$$

Features like **Exercise-Induced Angina ($exang$)**, **ST Depression ($oldpeak$)**, and **Number of Blocked Vessels ($ca$)** are weighted heavily based on real clinical cardiology studies.

### 2. Feature Contributions (Local Interpretability)
Standard ML models are "black boxes." CardioGuard AI addresses this by calculating a localized contribution score for each parameter relative to healthy baselines, weighted by the model's global feature importances:

$$\text{Contribution}_i = \frac{(\text{Patient Value}_i - \text{Baseline}_i)}{\text{Scale}_i} \times \text{Feature Importance}_i \times 100$$

This allows clinicians to see exactly *why* a patient is marked high-risk (e.g., *"Elevated ST depression increased the score by +15.5%"*).

---

## 🚀 Setting Up & Running Locally

### Step 1: Clone or Navigate to the Directory
Make sure you are in the project folder:
```bash
cd cardioguard_ai
```

### Step 2: Install Python Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Machine Learning Pipeline
Train the Random Forest classifier. This script will automatically call `generate_data.py` if the CSV dataset is not present, evaluate the model, and export the serialized assets:
```bash
python train_model.py
```
*Output you will see:*
- Precision, Recall, F1-Score (target accuracy around **~80%** with ROC-AUC **~0.83**).
- Feature Importance ranking (typically led by `oldpeak`, `ca`, and `age`).
- A saved file named `cardioguard_model.pkl`.

### Step 4: Boot up the Server
Start the Flask web backend:
```bash
python app.py
```
*The server will boot on `http://127.0.0.1:5000/`.*

### Step 5: Open the Dashboard
- In a web browser, navigate to `http://127.0.0.1:5000/`.
- Alternatively, you can double-click and open `index.html` directly in any web browser! The smart **dual-mode engine** will automatically load in sandbox-mode and run predictions using the high-fidelity embedded JavaScript risk compiler!

---

## 🩺 The Diagnostic Profiles (Test Cases)

You can select any of these profiles in the dashboard sidebar to see the AI model in action:
* **Sarah Jenkins (Low Risk)**: 34-year-old active female. All vitals are optimal. The risk score sits under **5%**, and the dashboard displays an encouraging lifestyle confirmation.
* **Robert Miller (Moderate Risk)**: 52-year-old male with mild, atypical chest pain and borderline blood pressure (138 mmHg). The risk score sits around **40%**, triggering early lifestyle warnings.
* **Maria Rodriguez (High Risk)**: 61-year-old female patient with high fasting blood sugar (diabetic indicator) and abnormal resting ECG results. The risk score is highly elevated, alerting the clinician to order metabolic screening.
* **Arthur Pendelton (High Critical Risk)**: 68-year-old male presenting asymptomatic but with 2 major blocked vessels under fluoroscopy and substantial exercise ST depression (2.8mm). The gauge shifts to **critical red (>80%)**, and the CDSS recommends an urgent Cardiology/Angiography referral.

---

## 🏆 Portfolio Highlights for AI Engineers

This project is an exceptional portfolio builder because it addresses key real-world engineering paradigms:
1. **Explainable AI (XAI)**: Demystifies predictions, turning raw classification probabilities into readable, visual, clinical drivers.
2. **Clinical Decision Support System (CDSS)**: Implements automated medical logic translating model weights and inputs into customized, patient-specific guidelines (consistent with AHA/ACC guidelines).
3. **Resilient Offline Architecture**: Leverages fallback logic (fetch error handling) to keep client-side apps fully operational even when server connections are severed, a vital standard in critical hospital systems.
4. **Clean Code & Modularity**: Demonstrates separation of concerns across dataset generation, modular ML training, REST API architecture, and a modern frontend SPA.

