import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===========================
# 1. LOAD DATA
# ===========================
print(">> LOAD DATA")
df = pd.read_csv("DATASET_HEALTH_LIFESTYLE.csv")
print(df.head())

# ===========================
# 2. PREPROCESSING
# ===========================
# Encoding gender
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# Buang kolom yang tidak relevan
X = df.drop(["id", "disease_risk"], axis=1)

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# 3. K-MEANS CLUSTERING
# ===========================
print("\n>> K-MEANS CLUSTERING")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters
print("Jumlah data per cluster:")
print(df["cluster"].value_counts())

# ===========================
# 4. PCA (2D Projection untuk analisis)
# ===========================
print("\n>> PCA REDUCTION TO 2D (untuk analisis, bukan visual)")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["pca_1"] = pca_result[:, 0]
df["pca_2"] = pca_result[:, 1]

print(df[["pca_1", "pca_2", "cluster"]].head())

# ===========================
# 5. ANALISIS CLUSTER
# ===========================
print("\n>> ANALISIS RATA-RATA TIAP CLUSTER")
cluster_summary = df.groupby("cluster").mean()
print(cluster_summary)

print("\n>> SELESAI — HASIL CLUSTERING SUDAH SIAP UNTUK LAPORAN")
