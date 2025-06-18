import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import os

#  Nettoyage du Texte
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

# Définir X et y pour les étapes suivantes
X = df['Cleaned_Resume']
y = df['Category_Encoded']

# --- Vectorisation TF-IDF ---
tfidf_vectorizer = TfidfVectorizer(
      max_features=5000,
      stop_words='english'
)

# Adapter et transformer les données textuelles
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Entraînement du Modèle
logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_tfidf, y)

nb_model = MultinomialNB()
nb_model.fit(X_tfidf, y)

# Sauvegarde
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Notebook/Analyse CV (Catégorie)"))

model_filename = os.path.join(output_dir, 'logistic_regression_model.joblib')
vectorizer_filename = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
label_encoder_filename = os.path.join(output_dir, 'label_encoder.joblib')

joblib.dump(logreg_model, model_filename)
joblib.dump(tfidf_vectorizer, vectorizer_filename)
joblib.dump(label_encoder, label_encoder_filename)
