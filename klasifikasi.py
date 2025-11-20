# ===========================
# BIG DATA - KLASIFIKASI PENYAKIT
# Menggunakan Logistic Regression, Random Forest, SVM
# Dengan SMOTE agar data seimbang
# ===========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# ===========================
# 1. LOAD DATA
# ===========================
print("===== LOAD DATA =====")
df = pd.read_csv("DATASET_HEALTH_LIFESTYLE.csv")
print(df.head())
print("\n===== INFO DATA =====")
print(df.info())

# ===========================
# 2. CEK NILAI KOSONG
# ===========================
print("\n===== CEK NILAI KOSONG =====")
print(df.isnull().sum())

# ===========================
# 3. ENCODING GENDER
# ===========================
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# ===========================
# 4. SPLIT DATA
# ===========================
X = df.drop(["disease_risk", "id"], axis=1)
y = df["disease_risk"]

# ===========================
# 5. NORMALISASI
# ===========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# 6. SMOTE UNTUK IMBALANCED DATA
# ===========================
print("\n===== SMOTE RESAMPLING =====")
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

print("Sebelum SMOTE:", y.value_counts())
print("Sesudah SMOTE:", y_resampled.value_counts())

# ===========================
# 7. TRAIN-TEST SPLIT
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ===========================
# 8. MODEL 1: LOGISTIC REGRESSION
# ===========================
print("\n===== LOGISTIC REGRESSION =====")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# ===========================
# 9. MODEL 2: RANDOM FOREST
# ===========================
print("\n===== RANDOM FOREST =====")
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ===========================
# 10. MODEL 3: SVM
# ===========================
print("\n===== SVM =====")
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ===========================
# 11. SELESAI
# ===========================
print("\n===== SEMUA MODEL SELESAI DI-RUN =====")
