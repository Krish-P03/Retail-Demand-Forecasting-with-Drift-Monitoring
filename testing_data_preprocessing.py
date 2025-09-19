import pandas as pd
import numpy as np
import joblib

df_test = pd.read_csv("test.csv")
df_test.columns = df_test.columns.str.strip()
df_test['Date'] = df_test['Date'].astype(str).str.strip()
df_test['Date'] = pd.to_datetime(df_test['Date'], dayfirst=True, errors='coerce')

if df_test['Date'].isna().sum() > 0:
    print(f"Warning: {df_test['Date'].isna().sum()} invalid dates found. Filling with earliest valid date.")
    df_test['Date'].fillna(df_test['Date'].min(), inplace=True)

df_test = df_test.reset_index(drop=True)

if df_test.empty:
    raise ValueError("processed_test.csv would be empty! Check test.csv for valid rows.")


df_test = df_test.sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
df_test['Year'] = df_test['Date'].dt.year
df_test['Month'] = df_test['Date'].dt.month
df_test['Week'] = df_test['Date'].dt.isocalendar().week
df_test['Day'] = df_test['Date'].dt.day
df_test['DayOfWeek'] = df_test['Date'].dt.dayofweek
df_test['IsMonthStart'] = df_test['Date'].dt.is_month_start.astype(int)
df_test['IsMonthEnd'] = df_test['Date'].dt.is_month_end.astype(int)
df_test['IsQuarterStart'] = df_test['Date'].dt.is_quarter_start.astype(int)
df_test['IsQuarterEnd'] = df_test['Date'].dt.is_quarter_end.astype(int)

if 'IsHoliday' in df_test.columns:
    df_test['IsHoliday'] = df_test['IsHoliday'].astype(int)
else:
    df_test['IsHoliday'] = 0

features = joblib.load("features.pkl")

for f in features:
    if f not in df_test.columns:
        df_test[f] = 0  

if df_test.empty:
    raise ValueError("After adding missing features, DataFrame is still empty!")

df_test.to_csv("processed_test.csv", index=False)
print("Test data preprocessed and saved as 'processed_test.csv'.")
