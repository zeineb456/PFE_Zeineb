import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

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

cat_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop('Attrition', axis=1)
y = df['Attrition']
feature_names = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

model_full = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_full.fit(X_train_full, y_train_full)

importances = model_full.feature_importances_
feat_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_features = feat_importances.head(15)['Feature'].tolist()
X_top = X[top_features]

scaler_top = StandardScaler()
X_top_scaled = scaler_top.fit_transform(X_top)

model_top = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)

model_top.fit(X_top_scaled, y)

joblib.dump(model_top, "Notebook\\Prédiction Employé\\modele_attrition_reduit.pkl")
joblib.dump(scaler_top, "Notebook\\Prédiction Employé\\scaler_attrition_reduit.pkl")
joblib.dump(top_features, "Notebook\\Prédiction Employé\\colonnes_utilisees_reduit.pkl")

