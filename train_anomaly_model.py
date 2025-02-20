import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib

# 🔹 Load Dataset
file_path = "vehicle_trajectory_dataset.csv"  # Update this path
df = pd.read_csv(file_path)

# 🔹 Print column names to check if they match the expected ones
print("Dataset Columns:", df.columns)

# 🔹 Step 1: Identify Ego Vehicle and Its Preceding Vehicle
ego_vehicle_id = df["Vehicle_ID"].unique()[0]  # Assuming first vehicle is ego
df_ego = df[df["Vehicle_ID"] == ego_vehicle_id]

# 🔹 Step 2: Extract Features for the Frontier Vehicle (Preceding Vehicle)
frontier_vehicle_ids = df_ego["Preceding"].unique()  # Get unique preceding vehicle IDs
df_frontier = df[df["Vehicle_ID"].isin(frontier_vehicle_ids)]  # Keep only those vehicles

# 🔹 Step 3: Use Correct Feature Names
features = ["v_Vel", "v_Acc", "Lane_ID"]  # Use actual column names
df_frontier = df_frontier[features]

# 🔹 Step 4: Verify if df_frontier is empty
if df_frontier.empty:
    print("⚠️ Warning: No frontier vehicles found. Check your filtering logic!")

# 🔹 Step 5: Normalize Data
scaler = MinMaxScaler()
df_frontier_scaled = pd.DataFrame(scaler.fit_transform(df_frontier), columns=df_frontier.columns)

# 🔹 Step 6: Train Isolation Forest for Anomaly Detection
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(df_frontier_scaled)

# 🔹 Step 7: Save Model and Scaler
joblib.dump(model, "frontier_anomaly_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model Training Completed: Anomalies in Frontier Vehicles Can Be Detected!")
