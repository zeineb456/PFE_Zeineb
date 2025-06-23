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
import tempfile

def setup_paths():
    """Configure paths in a cross-platform way"""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = SCRIPT_DIR.parent / "Notebook" / "Analyse CV (Catégorie)" / "ResumeCV.csv"
    OUTPUT_DIR = SCRIPT_DIR.parent / "Notebook" / "Analyse CV (Catégorie)"
    
    # Use environment variable for MLflow directory in CI
    if os.environ.get('GITHUB_WORKSPACE'):
        MLRUNS_DIR = Path(os.environ['GITHUB_WORKSPACE']) / "mlruns"
    else:
        MLRUNS_DIR = SCRIPT_DIR / "mlruns"
    
    # Create directories with proper permissions
    MLRUNS_DIR.mkdir(exist_ok=True, parents=True, mode=0o777)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    return DATA_DIR, OUTPUT_DIR, MLRUNS_DIR

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

def log_metrics(y_true, y_pred, model_name, class_names):
    """Log metrics and handle temporary files safely"""
    metrics = {
        f"{model_name}_accuracy": accuracy_score(y_true, y_pred),
        f"{model_name}_precision": precision_score(y_true, y_pred, average='weighted'),
        f"{model_name}_recall": recall_score(y_true, y_pred, average='weighted'),
        f"{model_name}_f1": f1_score(y_true, y_pred, average='weighted'),
        f"{model_name}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{model_name}_r2": r2_score(y_true, y_pred)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Classification report (saved to temporary file)
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        pd.DataFrame(report).transpose().to_json(tmp.name)
        mlflow.log_artifact(tmp.name)
    
    # Print metrics
    print(f"\n{model_name.upper()} Metrics:")
    for name, value in metrics.items():
        print(f"{name.split('_')[-1].title()}: {value:.4f}")

def main():
    # Setup paths and MLflow
    DATA_DIR, OUTPUT_DIR, MLRUNS_DIR = setup_paths()
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("Resume_Category_Classification")
    
    try:
        # Load data
        df = pd.read_csv(DATA_DIR)
        print("Data loaded successfully. Shape:", df.shape)
        
        # Clean text and encode categories
        df['Cleaned_Resume'] = df['Resume'].apply(clean_text)
        label_encoder = LabelEncoder()
        df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])
        class_names = label_encoder.classes_
        
        print("\nCategory Encoding:")
        for category, code in zip(class_names, label_encoder.transform(class_names)):
            print(f"- {category}: {code}")
        
        # Split data
        X = df['Cleaned_Resume']
        y = df['Category_Encoded']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        with mlflow.start_run():
            # Log parameters and tags
            mlflow.log_params({
                "max_features": 5000,
                "test_size": 0.2,
                "random_state": 42,
                "model_types": "logistic_regression,naive_bayes"
            })
            mlflow.set_tags({
                "project": "resume-classification",
                "team": "data-science",
                "version": "1.0"
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
            log_metrics(y_test, y_pred_logreg, "logistic_regression", class_names)
            mlflow.sklearn.log_model(logreg_model, "logistic_regression_model")
            
            # 2. Naive Bayes
            nb_model = MultinomialNB()
            nb_model.fit(X_train_tfidf, y_train)
            y_pred_nb = nb_model.predict(X_test_tfidf)
            log_metrics(y_test, y_pred_nb, "naive_bayes", class_names)
            mlflow.sklearn.log_model(nb_model, "naive_bayes_model")
            
            # Save preprocessing artifacts
            mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer")
            mlflow.sklearn.log_model(label_encoder, "label_encoder")
            
            # Feature importance (Logistic Regression)
            if hasattr(logreg_model, 'coef_'):
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                    feature_importance = pd.DataFrame({
                        'feature': tfidf_vectorizer.get_feature_names_out(),
                        'importance': logreg_model.coef_[0]
                    }).sort_values('importance', ascending=False)
                    feature_importance.to_csv(tmp.name, index=False)
                    mlflow.log_artifact(tmp.name)
            
            # Save models locally
            joblib.dump(logreg_model, OUTPUT_DIR / 'logistic_regression_model.joblib')
            joblib.dump(nb_model, OUTPUT_DIR / 'naive_bayes_model.joblib')
            joblib.dump(tfidf_vectorizer, OUTPUT_DIR / 'tfidf_vectorizer.joblib')
            joblib.dump(label_encoder, OUTPUT_DIR / 'label_encoder.joblib')
            
            print("\nTraining completed successfully!")
            
    except Exception as e:
        print(f"\nError during execution: {e}")
        if mlflow.active_run():
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
        raise

if __name__ == "__main__":
    main()
