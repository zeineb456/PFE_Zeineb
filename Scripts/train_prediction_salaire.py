import pandas as pd
import os
import kagglehub
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize MLflow
mlflow.set_experiment("Salary_Prediction_Neural_Network")

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns").replace("\\", "/"))

with mlflow.start_run():
    # --- Data Loading and Preparation ---
    path = kagglehub.dataset_download("mrsimple07/salary-prediction-data")
    files = os.listdir(path)
    csv_file = [f for f in files if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    # Load second dataset
    path2 = kagglehub.dataset_download("rkiattisak/salaly-prediction-for-beginer")
    files2 = os.listdir(path2)
    csv_file2 = [f for f in files2 if f.endswith('.csv')][0]
    df2 = pd.read_csv(os.path.join(path2, csv_file2))

    # Data cleaning and preprocessing
    df.drop(columns=["Location"], inplace=True)
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Experience": "Years of Experience",
        "Job_Title": "Job Title",
        "Education": "Education Level"
    }, inplace=True)
    df2.columns = df2.columns.str.strip()

    # Combine datasets
    df1 = df[["Education Level", "Years of Experience", "Job Title", "Age", "Gender", "Salary"]]
    df2 = df2[["Education Level", "Years of Experience", "Job Title", "Age", "Gender", "Salary"]]
    df_combined = pd.concat([df, df2], ignore_index=True)
    df_combined.dropna(how='all', inplace=True)

    # Outlier detection and removal
    Q1 = df_combined['Salary'].quantile(0.25)
    Q3 = df_combined['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    df_combined = df_combined[(df_combined['Salary'] >= (Q1 - 1.5 * IQR)) & 
                             (df_combined['Salary'] <= (Q3 + 1.5 * IQR))]

    # Education level standardization
    df_combined['Education Level'] = df_combined['Education Level'].replace({
        'Bachelor': "Bachelor's",
        'Master': "Master's"
    })

    # Log dataset information
    mlflow.log_param("dataset_size", len(df_combined))
    mlflow.log_param("salary_mean", df_combined['Salary'].mean())
    mlflow.log_param("salary_std", df_combined['Salary'].std())

    # Prepare features and target
    X = df_combined.drop("Salary", axis=1)
    y = df_combined["Salary"].values.reshape(-1, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define preprocessing
    num_cols = ["Age", "Years of Experience"]
    cat_onehot = ["Gender", "Job Title"]
    cat_ordinal = ["Education Level"]
    education_order = [['High School', "Bachelor's", "Master's", 'PhD']]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("edu", OrdinalEncoder(categories=education_order), cat_ordinal),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_onehot)
    ])

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Target scaling
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Log preprocessing details
    mlflow.log_param("num_features", len(num_cols))
    mlflow.log_param("cat_features", len(cat_onehot + cat_ordinal))
    mlflow.log_param("total_features", X_train_processed.shape[1])

    # --- Neural Network Architecture ---
    input_dim = X_train_processed.shape[1]
    hidden_dim = 64
    output_dim = 1

    # Log model parameters
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 1000)

    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))

    # Training loop
    learning_rate = 0.01
    epochs = 1000
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Forward pass
        Z1 = np.dot(X_train_processed, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        y_pred_train = Z2

        # Calculate loss
        train_loss = mse_loss(y_train_scaled, y_pred_train)
        train_losses.append(train_loss)

        # Test set evaluation
        Z1_test = np.dot(X_test_processed, W1) + b1
        A1_test = relu(Z1_test)
        Z2_test = np.dot(A1_test, W2) + b2
        y_pred_test = Z2_test
        test_loss = mse_loss(y_test_scaled, y_pred_test)
        test_losses.append(test_loss)

        # Backpropagation
        dZ2 = 2 * (y_pred_train - y_train_scaled) / y_train_scaled.shape[0]
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_train_processed.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Log metrics every 100 epochs
        if epoch % 100 == 0:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)

    # Final evaluation
    y_pred_test_unscaled = scaler_y.inverse_transform(Z2_test)
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test_unscaled, y_pred_test_unscaled)
    mae = mean_absolute_error(y_test_unscaled, y_pred_test_unscaled)
    r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)

    # Log final metrics
    mlflow.log_metrics({
        "final_mse": mse,
        "final_mae": mae,
        "final_r2": r2,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1]
    })

    # Print metrics
    print("\nModel Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # --- Save Artifacts ---
    # Save weights and biases
    weights = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    joblib.dump(weights, "neural_network_weights.pkl")
    mlflow.log_artifact("neural_network_weights.pkl")

    # Save preprocessor and scaler
    mlflow.sklearn.log_model(preprocessor, "preprocessor")
    mlflow.sklearn.log_model(scaler_y, "target_scaler")

    # Save training history plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig("training_history.png")
    mlflow.log_artifact("training_history.png")

    # Save locally
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Notebook/Prédiction Salaire"))
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(weights, os.path.join(output_dir, 'neural_network_weights.pkl'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))
    
    print("\nTraining complete. All artifacts saved.")