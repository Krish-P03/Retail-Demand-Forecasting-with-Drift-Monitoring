import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("train.csv")

# Sort and Reset Index
df = df.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
df['IsQuarterStart'] = df['Date'].dt.is_quarter_start.astype(int)
df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
if 'IsHoliday' in df.columns:
    df['IsHoliday'] = df['IsHoliday'].astype(int)
else:
    df['IsHoliday'] = 0
    
# Lag Features (previous weeks)
df = df.sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
df['Lag_1'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1)
df['Lag_2'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(2)
df['Lag_3'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(3)

# Rolling Features
df['Roll_Mean_4'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(4).mean()
df['Roll_Mean_8'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(8).mean()

# Drop rows with NaN after lag/rolling
df = df.dropna().reset_index(drop=True)

# Save Cleaned/Processed Data
df.to_csv("cleaned.csv", index=False)
print("Data preprocessing completed.")
