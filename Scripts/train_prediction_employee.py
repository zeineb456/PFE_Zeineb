import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

# Initialize MLflow
mlflow.set_experiment("IBM_HR_Attrition_Prediction")

def log_metrics(y_true, y_pred, y_proba=None, prefix=""):
    """Log evaluation metrics to MLflow"""
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),

    }
    
    if y_proba is not None:
        metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    
    mlflow.log_metrics(metrics)
    
    # Log confusion matrix as artifact
    cm = confusion_matrix(y_true, y_pred)
    cm_path = f"{prefix}confusion_matrix.txt"
    np.savetxt(cm_path, cm, fmt='%d')
    mlflow.log_artifact(cm_path)
    
    # Print metrics
    print(f"\n{prefix} Model Performance:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
# Crée un dossier local pour les artefacts
mlruns_path = os.path.abspath("mlruns")

# Conversion du chemin pour compatibilité Windows/Linux
if os.name == "nt":  # Windows
    tracking_uri = "file:///" + mlruns_path.replace("\\", "/")
else:  # Linux/Mac
    tracking_uri = "file://" + mlruns_path

mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run():
    # --- Data Preparation ---
    # Télécharger depuis Kaggle
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")

    # Localiser le fichier CSV
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                break

    # Charger dans un DataFrame
    df = pd.read_csv(csv_path)

    # Data preprocessing
    mappings = {
        'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'},
        'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
        'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'WorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
    }
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df.drop(['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)

    # Log dataset information
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("attrition_rate", df['Attrition'].mean())

    # Feature engineering
    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    feature_names = X.columns

    # Train-test split
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    # --- Full Model Training ---
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    model_full = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_full.fit(X_train_scaled, y_train_full)
    
    # Evaluate full model
    y_pred_full = model_full.predict(X_test_scaled)
    y_proba_full = model_full.predict_proba(X_test_scaled)
    log_metrics(y_test_full, y_pred_full, y_proba_full, "full_")
    
    # Feature importance analysis
    importances = model_full.feature_importances_
    feat_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Log top features
    top_features = feat_importances.head(15)['Feature'].tolist()
    mlflow.log_param("top_features", top_features)
    
    # Save feature importance plot
    feat_importances.head(15).to_csv("feature_importance.csv")
    mlflow.log_artifact("feature_importance.csv")

    # --- Reduced Model Training ---
    X_top_train = X_train_full[top_features]
    X_top_test = X_test_full[top_features]
    
    scaler_top = StandardScaler()
    X_top_train_scaled = scaler_top.fit_transform(X_top_train)
    X_top_test_scaled = scaler_top.transform(X_top_test)

    model_top = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model_top.fit(X_top_train_scaled, y_train_full)
    
    # Evaluate reduced model
    y_pred_top = model_top.predict(X_top_test_scaled)
    y_proba_top = model_top.predict_proba(X_top_test_scaled)
    log_metrics(y_test_full, y_pred_top, y_proba_top, "reduced_")

    # --- Log Models and Artifacts ---
    # Log full model
    mlflow.sklearn.log_model(model_full, "full_model")
    
    # Log reduced model
    mlflow.sklearn.log_model(model_top, "reduced_model")
    mlflow.sklearn.log_model(scaler_top, "reduced_scaler")
    
    # Log parameters of reduced model
    mlflow.log_param("reduced_features_count", len(top_features))
    mlflow.log_param("class_weight", "balanced")

    # Save artifacts locally
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Notebook/Prédiction Employé"))
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model_top, os.path.join(output_dir, "modele_attrition_reduit.pkl"))
    joblib.dump(scaler_top, os.path.join(output_dir, "scaler_attrition_reduit.pkl"))
    joblib.dump(top_features, os.path.join(output_dir, "colonnes_utilisees_reduit.pkl"))
    
    # Log local files as artifacts
    mlflow.log_artifact(os.path.join(output_dir, "modele_attrition_reduit.pkl"))
    mlflow.log_artifact(os.path.join(output_dir, "scaler_attrition_reduit.pkl"))
    mlflow.log_artifact(os.path.join(output_dir, "colonnes_utilisees_reduit.pkl"))
    
    print("\nTraining complete. Models and artifacts logged to MLflow and saved locally.")