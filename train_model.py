import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def train_and_save_model():
    data_path = "heart_disease_data.csv"
    
    # Generate data if not present
    if not os.path.exists(data_path):
        print(f"Dataset '{data_path}' not found. Generating it now...")
        from generate_data import generate_heart_disease_dataset
        df = generate_heart_disease_dataset(n_samples=1200)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
        
    print(f"Loaded clinical dataset with {len(df)} records.")
    
    # Split features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale continuous features
    scaler = StandardScaler()
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Create copies to avoid warnings
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n================ MODEL EVALUATION METRICS ================")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Negatives (Healthy): {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives (High Risk): {cm[1][1]}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("==========================================================")
    
    # Feature Importances
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importances:")
    for feat, imp in feat_imp:
        print(f" - {feat:<10}: {imp:.4f}")
        
    # Save Model and Scaler
    artifacts = {
        'model': model,
        'scaler': scaler,
        'continuous_cols': continuous_cols,
        'feature_names': list(feature_names),
        'feature_importances': {feat: float(imp) for feat, imp in feat_imp}
    }
    
    model_filename = "cardioguard_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(artifacts, f)
        
    print(f"\nModel and scaler saved successfully to '{model_filename}'!")

if __name__ == "__main__":
    train_and_save_model()
