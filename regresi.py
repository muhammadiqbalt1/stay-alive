# regresi.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Ubah path jika perlu. Jika file ada di folder yg sama, cukup nama file:
CSV_PATH = "DATASET_HEALTH_LIFESTYLE.csv"
# atau di environment lain gunakan "/mnt/data/DATASET_HEALTH_LIFESTYLE.csv"

# 1. Load data
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print(df.head())

# 2. Pilih fitur & target (target: kolom 'cholesterol')
# Pastikan kolom 'cholesterol' ada di dataset Anda
y = df["cholesterol"]
X = df.drop(["cholesterol", "id", "disease_risk"], axis=1)  # buang kolom yg tidak dipakai

# Encode gender
X["gender"] = X["gender"].map({"Male":1, "Female":0})

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scaling (opsional untuk regresi tree tidak wajib, tapi baik untuk linear)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)
print("\nLinear Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("Linear Regression R2:", r2_score(y_test, pred_lr))

# 6. Model 2: Random Forest Regressor
rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # tree-based tidak perlu scaling
pred_rf = rf.predict(X_test)
print("\nRandom Forest RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))
print("Random Forest R2:", r2_score(y_test, pred_rf))

# 7. Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance (top 10):\n", importances.head(10))

# 8. (Opsional) Simpan model jika mau
# import joblib
# joblib.dump(rf, "rf_regression_cholesterol.pkl")
