# resume_classifier.py
import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, 
                            mean_squared_error, r2_score,
                            classification_report)
import os
import mlflow
import mlflow.sklearn
from pathlib import Path

# Configure paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "Notebook" / "Analyse CV (Catégorie)" / "ResumeCV.csv"
OUTPUT_DIR = SCRIPT_DIR.parent / "Notebook" / "Analyse CV (Catégorie)"
MLRUNS_DIR = SCRIPT_DIR / "mlruns"

# Ensure directories exist
MLRUNS_DIR.mkdir(exist_ok=True, parents=True, mode=0o777)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Load and prepare data
try:
    df = pd.read_csv(DATA_DIR)
    print("Data loaded successfully. Shape:", df.shape)
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Clean text and encode categories
df['Cleaned_Resume'] = df['Resume'].apply(clean_text)
label_encoder = LabelEncoder()
df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])
class_names = label_encoder.classes_

# Show category mapping
print("\nCategory Encoding:")
for category, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"- {category}: {code}")

# Split data
X = df['Cleaned_Resume']
y = df['Category_Encoded']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# MLflow setup
mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
mlflow.set_experiment("Resume_Category_Classification")
mlflow.set_experiment_tags({
    "project": "resume-classification",
    "team": "data-science",
    "version": "1.0"
})

# Metrics function
def log_metrics(y_true, y_pred, model_name):
    metrics = {
        f"{model_name}_accuracy": accuracy_score(y_true, y_pred),
        f"{model_name}_precision": precision_score(y_true, y_pred, average='weighted'),
        f"{model_name}_recall": recall_score(y_true, y_pred, average='weighted'),
        f"{model_name}_f1": f1_score(y_true, y_pred, average='weighted'),
        f"{model_name}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{model_name}_r2": r2_score(y_true, y_pred)
    }
    
    # Log metrics
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    mlflow.log_dict(report, f"{model_name}_classification_report.json")
    
    # Print metrics
    print(f"\n{model_name.upper()} Metrics:")
    for name, value in metrics.items():
        print(f"{name.split('_')[-1].title()}: {value:.4f}")

# Main training and logging
try:
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "max_features": 5000,
            "test_size": 0.2,
            "random_state": 42,
            "model_types": "logistic_regression,naive_bayes"
        })
        
        # 1. Logistic Regression
        logreg_model = LogisticRegression(random_state=42, max_iter=1000)
        logreg_model.fit(X_train_tfidf, y_train)
        y_pred_logreg = logreg_model.predict(X_test_tfidf)
        
        # Cross-validation
        cv_scores = cross_val_score(logreg_model, X_train_tfidf, y_train, cv=5)
        mlflow.log_metrics({
            "logreg_cv_accuracy_mean": cv_scores.mean(),
            "logreg_cv_accuracy_std": cv_scores.std()
        })
        
        # Log metrics and model
        log_metrics(y_test, y_pred_logreg, "logistic_regression")
        mlflow.sklearn.log_model(logreg_model, "logistic_regression_model")
        
        # 2. Naive Bayes
        nb_model = MultinomialNB()
        nb_model.fit(X_train_tfidf, y_train)
        y_pred_nb = nb_model.predict(X_test_tfidf)
        
        log_metrics(y_test, y_pred_nb, "naive_bayes")
        mlflow.sklearn.log_model(nb_model, "naive_bayes_model")
        
        # Save preprocessing artifacts
        mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer")
        mlflow.sklearn.log_model(label_encoder, "label_encoder")
        
        # Feature importance (Logistic Regression)
        if hasattr(logreg_model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': tfidf_vectorizer.get_feature_names_out(),
                'importance': logreg_model.coef_[0]
            }).sort_values('importance', ascending=False)
            
            importance_file = OUTPUT_DIR / "feature_importance.csv"
            feature_importance.to_csv(importance_file, index=False)
            mlflow.log_artifact(importance_file)
        
        # Save models locally
        joblib.dump(logreg_model, OUTPUT_DIR / 'logistic_regression_model.joblib')
        joblib.dump(nb_model, OUTPUT_DIR / 'naive_bayes_model.joblib')
        joblib.dump(tfidf_vectorizer, OUTPUT_DIR / 'tfidf_vectorizer.joblib')
        joblib.dump(label_encoder, OUTPUT_DIR / 'label_encoder.joblib')
        
        print("\nTraining completed successfully!")
        
except Exception as e:
    print(f"\nError during MLflow run: {e}")
    if 'mlflow.active_run' in globals():
        mlflow.log_param("error", str(e))
    raise