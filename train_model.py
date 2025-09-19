import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.lightgbm

# Load Cleaned Data
df = pd.read_csv("cleaned.csv")

# Features and Target
target = 'Weekly_Sales'
features = [col for col in df.columns if col not in ['Weekly_Sales','Date']]
X = df[features]
y = df[target]

# Train/Validation/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=False
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
)

# MLflow Setup
mlflow.set_experiment("walmart_sales_forecast")
with mlflow.start_run(run_name="lgbm_model") as run:
    #Define LightGBM Model
    model = LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        n_estimators=1000,
        random_state=42
    )
    #Train with Callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=50)
        ]
    )
    #Validation Metrics
    y_val_pred = model.predict(X_val)
    y_val_true = np.array(y_val)
    rmse_val = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    mae_val = mean_absolute_error(y_val_true, y_val_pred)
    r2_val = r2_score(y_val_true, y_val_pred)
    print(f"Validation RMSE: {rmse_val:.2f}")
    print(f"Validation MAE: {mae_val:.2f}")
    print(f"Validation R2: {r2_val:.4f}")
    #  Test Metrics
    y_test_pred = model.predict(X_test)
    y_test_true = np.array(y_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae_test = mean_absolute_error(y_test_true, y_test_pred)
    r2_test = r2_score(y_test_true, y_test_pred)
    print(f"Test RMSE: {rmse_test:.2f}")
    print(f"Test MAE: {mae_test:.2f}")
    print(f"Test R2: {r2_test:.4f}")
    
    #  MLflow Logging
    mlflow.log_params({
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 1000,
        "objective": "regression"
    })
    
    mlflow.log_metrics({
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "r2_val": r2_val,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test
    })
    
    mlflow.lightgbm.log_model(model, artifact_path="lgbm_model")
    
    # Save Model Locally
    joblib.dump(model, "lgbm_model.pkl")
    print("Model saved as 'lgbm_model.pkl'")
    
    joblib.dump(features, "features.pkl")
    print("Feature list saved as 'features.pkl'")
