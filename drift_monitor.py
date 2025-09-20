import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

train_data = pd.read_csv("cleaned.csv")
test_data = pd.read_csv("processed_test.csv")

target_column = "Weekly_Sales"
if target_column in test_data.columns:
    test_data = test_data.drop(columns=[target_column])

dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(reference_data=train_data, current_data=test_data)

dashboard.save("drift_report.html")
print("Drift monitoring report")
