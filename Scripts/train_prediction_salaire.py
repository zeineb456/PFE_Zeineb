import pandas as pd
import os
import kagglehub
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
import joblib


# Fonctions d'activation
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


path = kagglehub.dataset_download("mrsimple07/salary-prediction-data")
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Charger la 2e dataset
path2 = kagglehub.dataset_download("rkiattisak/salaly-prediction-for-beginer")
print("Path to second dataset files:", path2)

# Liste les fichiers CSV dans le dossier
files2 = os.listdir(path2)
csv_file2 = [f for f in files2 if f.endswith('.csv')][0]

# Charger la seconde dataset
df2 = pd.read_csv(os.path.join(path2, csv_file2))

df.drop(columns=["Location"], inplace=True)

df.columns = df.columns.str.strip()
df.rename(columns={
    "Experience": "Years of Experience",
    "Job_Title": "Job Title",
    "Education": "Education Level"
}, inplace=True)

df2.columns = df2.columns.str.strip()

df1 = df[["Education Level", "Years of Experience", "Job Title", "Age", "Gender", "Salary"]]
df2 = df2[["Education Level", "Years of Experience", "Job Title", "Age", "Gender", "Salary"]]

df_combined = pd.concat([df, df2], ignore_index=True)
df_combined.dropna(how='all', inplace=True)

# Calcul des quartiles et de l'IQR
Q1 = df_combined['Salary'].quantile(0.25)
Q3 = df_combined['Salary'].quantile(0.75)
IQR = Q3 - Q1

# Détection des outliers
outliers = df_combined[(df_combined['Salary'] < (Q1 - 1.5 * IQR)) | (df_combined['Salary'] > (Q3 + 1.5 * IQR))]

df_combined = df_combined.drop(index=1259)

df_combined['Education Level'] = df_combined['Education Level'].replace({
    'Bachelor': "Bachelor's",
    'Master': "Master's"
})

# Sélection des colonnes numériques
num_cols = ['Age', 'Years of Experience', 'Salary']
df_corr = df_combined[num_cols]

# Copie du dataframe
df_clean = df_combined.copy()

X = df_clean.drop("Salary", axis=1)
y = df_clean["Salary"].values.reshape(-1, 1)

# Colonnes
num_cols = ["Age", "Years of Experience"]
cat_onehot = ["Gender", "Job Title"]
cat_ordinal = ["Education Level"]

# Ordre logique pour Education
education_order = [['High School', "Bachelor's", "Master's", 'PhD']]

# Encodage
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("edu", OrdinalEncoder(categories=education_order), cat_ordinal),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_onehot)
])

# Transformation
X_processed = preprocessor.fit_transform(X)

# Normalisation cible
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Architecture du réseau
input_dim = X_processed.shape[1]
hidden_dim = 64
output_dim = 1

# Initialisation des poids
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

# Entraînement
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    Z1 = np.dot(X_processed, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    y_pred = Z2

    loss = mse_loss(y_scaled, y_pred)

    # Rétropropagation
    dZ2 = 2 * (y_pred - y_scaled) / y_scaled.shape[0]
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_processed.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Mise à jour des poids
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
# Poids et biais du réseau
joblib.dump(W1, 'Notebook\\Prédiction Salaire\\W1_model.pkl')
joblib.dump(b1, 'Notebook\\Prédiction Salaire\\b1_model.pkl')
joblib.dump(W2, 'Notebook\\Prédiction Salaire\\W2_model.pkl')
joblib.dump(b2, 'Notebook\\Prédiction Salaire\\b2_model.pkl')

# Préprocesseur complet
joblib.dump(preprocessor, 'Notebook\\Prédiction Salaire\\preprocessor.pkl')

# Scaler pour la cible
joblib.dump(scaler_y, 'Notebook\\Prédiction Salaire\\scaler_y.pkl')



