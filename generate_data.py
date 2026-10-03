import pandas as pd
import numpy as np

def generate_heart_disease_dataset(n_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Age (29 to 80)
    age = np.random.randint(30, 78, size=n_samples)
    
    # 2. Sex (0 = Female, 1 = Male)
    sex = np.random.binomial(1, 0.65, size=n_samples)
    
    # 3. Chest Pain Type (cp): 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal, 3 = Asymptomatic
    # Higher chance of asymptomatic (3) or typical (0) being high risk
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.15, 0.20, 0.25, 0.40])
    
    # 4. Resting Blood Pressure (trestbps): 94 to 200 mmHg
    trestbps = np.random.normal(130, 18, size=n_samples).astype(int)
    trestbps = np.clip(trestbps, 94, 200)
    
    # 5. Cholesterol (chol): 126 to 564 mg/dl
    chol = np.random.normal(240, 45, size=n_samples).astype(int)
    chol = np.clip(chol, 126, 500)
    
    # 6. Fasting Blood Sugar (fbs): (1 = > 120 mg/dl, 0 = <= 120 mg/dl)
    fbs_prob = 0.15 * (age / 50.0)
    fbs_prob = np.clip(fbs_prob, 0.05, 0.45)
    fbs = np.random.binomial(1, fbs_prob)
    
    # 7. Resting ECG (restecg): 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.50, 0.45, 0.05])
    
    # 8. Max Heart Rate Achieved (thalach): 71 to 202 bpm (decreases with age)
    thalach = (200 - 0.7 * age + np.random.normal(0, 12, size=n_samples)).astype(int)
    thalach = np.clip(thalach, 71, 202)
    
    # 9. Exercise Induced Angina (exang): 1 = Yes, 0 = No
    exang_prob = 0.2 + 0.3 * (cp == 3) + 0.1 * (age > 55)
    exang_prob = np.clip(exang_prob, 0.05, 0.90)
    exang = np.random.binomial(1, exang_prob)
    
    # 10. ST depression induced by exercise (oldpeak): 0.0 to 6.2
    oldpeak = np.random.exponential(scale=1.0, size=n_samples)
    oldpeak = np.clip(oldpeak, 0.0, 6.2)
    # Round to 1 decimal place
    oldpeak = np.round(oldpeak, 1)
    
    # 11. Slope of peak exercise ST segment: 0 = Upsloping, 1 = Flat, 2 = Downsloping
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.45, 0.45, 0.10])
    
    # 12. Number of major vessels (ca): 0 to 4
    ca = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.55, 0.22, 0.12, 0.07, 0.04])
    
    # 13. Thalassemia (thal): 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect
    thal = np.random.choice([1, 2, 3], size=n_samples, p=[0.55, 0.10, 0.35])
    
    # --- Now we calculate the probability of heart disease using a logit model to make correlations realistic ---
    # Logit formula: z = beta_0 + sum(beta_i * X_i)
    z = (
        -4.5
        + 0.04 * (age - 50)               # Older age increases risk
        + 0.8 * sex                       # Males have higher risk in general
        + 0.5 * (cp == 3)                 # Asymptomatic chest pain is highly correlated with severe underlying CAD
        + 0.3 * (cp == 0)                 # Typical angina
        + 0.015 * (trestbps - 120)        # Elevated BP increases risk
        + 0.005 * (chol - 200)            # High cholesterol increases risk
        + 0.3 * fbs                       # Diabetes/High FBS increases risk
        + 0.4 * (restecg > 0)             # ECG abnormalities increase risk
        - 0.02 * (thalach - 150)          # Higher peak heart rate is protective/correlated with fitness
        + 1.2 * exang                     # Exercise induced angina is a strong predictor
        + 0.8 * oldpeak                   # High ST depression is a strong predictor
        + 0.5 * (slope == 1)              # Flat slope is higher risk
        + 0.9 * ca                        # More blocked vessels = much higher risk
        + 1.0 * (thal == 3)               # Reversible defect = high risk
    )
    
    # Sigmoid function to get probabilities
    prob = 1 / (1 + np.exp(-z))
    
    # Target (1 = Heart Disease, 0 = Healthy)
    target = np.random.binomial(1, prob)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic clinical heart disease dataset...")
    df = generate_heart_disease_dataset(n_samples=1200)
    df.to_csv("heart_disease_data.csv", index=False)
    print(f"Dataset generated successfully! Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
    print("\nFirst few rows:")
    print(df.head())
