import pandas as pd
import joblib

model = joblib.load("lgbm_model.pkl")
features = joblib.load("features.pkl")

df_test = pd.read_csv("processed_test.csv")
df_test.columns = df_test.columns.str.strip()
if df_test.empty:
    raise ValueError("processed_test.csv is empty! Cannot predict.")
for f in features:
    if f not in df_test.columns:
        df_test[f] = 0
X_test = df_test[features]

if X_test.empty:
    raise ValueError("X_test is empty. Check processed_test.csv and features.pkl")

predictions = model.predict(X_test)
df_test['Predicted_Sales'] = predictions

df_test.to_csv("predictions.csv", index=False)
print("Predictions saved as 'predictions.csv'!")
