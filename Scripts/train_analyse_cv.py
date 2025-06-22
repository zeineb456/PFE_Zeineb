import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import os
import mlflow
import mlflow.sklearn

# Nettoyage du Texte
def clean_text(text):
    text = str(text).lower() # mettre en minuscule
    text = text.translate(str.maketrans('', '', string.punctuation)) # Enlever ponctuation
    text = re.sub(r'\d+', '', text) # Enlever les nombres
    text = ' '.join(text.split()) # Enlever espaces superflus
    return text

DATA_DIR = os.path.join(os.path.dirname(__file__), "../Notebook/Analyse CV (Catégorie)/ResumeCV.csv")
DATA_DIR = os.path.abspath(DATA_DIR)

df = pd.read_csv(DATA_DIR)

categories = sorted(df['Category'].unique())
category_counts = df['Category'].value_counts()

df['Cleaned_Resume'] = df['Resume'].apply(clean_text)

# Encodage des Étiquettes
label_encoder = LabelEncoder()
df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])
class_names = label_encoder.classes_
for category, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"- {category}: {code}")
print(df[['Category', 'Category_Encoded']].head().to_markdown(index=False, numalign="left", stralign="left"))

# Split data into train and test sets
X = df['Cleaned_Resume']
y = df['Category_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Vectorisation TF-IDF ---
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

# Adapter et transformer les données textuelles
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Set up MLflow experiment
mlflow.set_experiment("Resume_Category_Classification")

# Function to calculate and log metrics
def log_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mlflow.log_metric(f"{model_name}_accuracy", accuracy)
    mlflow.log_metric(f"{model_name}_precision", precision)
    mlflow.log_metric(f"{model_name}_recall", recall)
    mlflow.log_metric(f"{model_name}_f1", f1)
    mlflow.log_metric(f"{model_name}_rmse", rmse)
    mlflow.log_metric(f"{model_name}_r2", r2)
    
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}\n")

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("test_size", 0.2)
    
    # Entraînement du Modèle Logistic Regression
    logreg_model = LogisticRegression(random_state=42, max_iter=1000)
    logreg_model.fit(X_train_tfidf, y_train)
    y_pred_logreg = logreg_model.predict(X_test_tfidf)
    
    # Log Logistic Regression metrics
    log_metrics(y_test, y_pred_logreg, "logistic_regression")
    
    # Log Logistic Regression model
    mlflow.sklearn.log_model(logreg_model, "logistic_regression_model")
    
    # Entraînement du Modèle Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    
    # Log Naive Bayes metrics
    log_metrics(y_test, y_pred_nb, "naive_bayes")
    
    # Log Naive Bayes model
    mlflow.sklearn.log_model(nb_model, "naive_bayes_model")
    
    # Log the vectorizer and label encoder
    mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer")
    mlflow.sklearn.log_model(label_encoder, "label_encoder")
    
    # Log feature importance (for logistic regression)
    if hasattr(logreg_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': tfidf_vectorizer.get_feature_names_out(),
            'importance': logreg_model.coef_[0]
        })
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
    
    # Save artifacts locally as well
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Notebook/Analyse CV (Catégorie)"))
    
    model_filename = os.path.join(output_dir, 'logistic_regression_model.joblib')
    vectorizer_filename = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
    label_encoder_filename = os.path.join(output_dir, 'label_encoder.joblib')
    
    joblib.dump(logreg_model, model_filename)
    joblib.dump(tfidf_vectorizer, vectorizer_filename)
    joblib.dump(label_encoder, label_encoder_filename)
    
    # Log the local files as artifacts
    mlflow.log_artifact(model_filename)
    mlflow.log_artifact(vectorizer_filename)
    mlflow.log_artifact(label_encoder_filename)