import pandas as pd
import joblib

# Load trained model
model = joblib.load("lgbm_model.pkl")

# Load training features
features = joblib.load("features.pkl")

# Load preprocessed test data
df_test = pd.read_csv("processed_test.csv")
df_test.columns = df_test.columns.str.strip()

# Safety check: DataFrame is not empty
if df_test.empty:
    raise ValueError("processed_test.csv is empty! Cannot predict.")

# Ensure all training features exist
for f in features:
    if f not in df_test.columns:
        df_test[f] = 0

# Select features in correct order
X_test = df_test[features]

# Safety check: X_test is valid
if X_test.empty:
    raise ValueError("X_test is empty. Check processed_test.csv and features.pkl")

# Make predictions
predictions = model.predict(X_test)
df_test['Predicted_Sales'] = predictions

# Save predictions
df_test.to_csv("predictions.csv", index=False)
print("Predictions saved as 'predictions.csv'!")
