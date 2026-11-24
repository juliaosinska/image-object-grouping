import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# WŁASNE: K-means on two chosen descriptors
def run_kmeans(df, feature_x, feature_y, k):
    # 1. data for clustering: only the two chosen columns (N x 2)
    X = df[[feature_x, feature_y]].to_numpy()

    # 2. standardization: mean and standard deviation per column + transform to (x - mean) / std dev
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. K-means on the standardized data
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    # 4. centroids back in the original units (to draw them on the same scatter plot)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # 5. comparison with the real categories (only AFTER the clustering)
    comparison = pd.crosstab(clusters, df["category"], rownames=["cluster"], colnames=["category"])

    return clusters, centroids, comparison
