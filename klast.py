import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
df = pd.read_csv("Live.csv")
drop_cols = [
    'status_id',
    'status_published',
    'Column1',
    'Column2',
    'Column3',
    'Column4'
]
df.drop(columns=[c for c in drop_cols if c in df.columns],
        inplace=True)
df = pd.get_dummies(df, drop_first=True)
df = df.loc[:, df.var() > 0.01]
print(df.isnull().sum())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
wcss = []
K = range(1, 15)
for k in K:
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        max_iter=500,
        n_init=20,
        random_state=42
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,6))
plt.plot(
    K,
    wcss,
    marker='o',
    linewidth=3,
    markersize=10,
    color='blue'
)
plt.xticks(K)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('Elbow Method For Optimal K', fontsize=16)
plt.xlabel('Number of Clusters (K)', fontsize=13)
plt.ylabel('WCSS', fontsize=13)
plt.axvline(x=3, color='red', linestyle='--', label='Possible Elbow')
plt.legend()
plt.show()
optimal_k = 3
kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    max_iter=500,
    n_init=20,
    random_state=42
)
y_kmeans = kmeans.fit_predict(X_scaled)
df['Cluster'] = y_kmeans
print("\nCluster Counts:")
print(df['Cluster'].value_counts())
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
viz_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Cluster': y_kmeans
})
plt.figure(figsize=(9,7))
sns.scatterplot(
    data=viz_df,
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    palette='Set1',
    s=120
)
plt.title('K-Means Cluster Visualization', fontsize=12)
plt.show()
print("\nCluster Centers:")
print(kmeans.cluster_centers_)