import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv("customers.csv")
print("First 5 rows:")
print(df.head())
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
print("\nCluster Centers:")
print(centers)
print("\nUpdated Dataset:")
print(df.head())
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["cluster"],
    palette="Set1",
    s=100
)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    s=300,
    marker="X",
    label="Centroids"
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()
df.to_csv("customers_clustered.csv", index=False)
print("\nClustered dataset saved as customers_clustered.csv")
print("\nCustomers in each cluster:")
print(df["cluster"].value_counts())
score = silhouette_score(X, df["cluster"])
print("\nSilhouette Score:", round(score, 2))
avg_spending = df.groupby("cluster")["Spending Score (1-100)"].mean()
print("\nAverage Spending Score per Cluster:")
print(avg_spending)
avg_income = df.groupby("cluster")["Annual Income (k$)"].mean()
print("\nAverage Annual Income per Cluster:")
print(avg_income)
plt.figure(figsize=(7, 5))
sns.countplot(x=df["cluster"], palette="Set2")
plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.grid(True)
plt.show()
for i in range(k):
    print(f"\nCluster {i} Details:")
    cluster_data = df[df["cluster"] == i]
    print(cluster_data.describe())
summary = df.groupby("cluster").mean(numeric_only=True)
summary.to_csv("cluster_summary.csv")
print("\nCluster summary saved as cluster_summary.csv")