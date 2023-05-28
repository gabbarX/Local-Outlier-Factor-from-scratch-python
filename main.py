from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the glass dataset
glass_df = pd.read_csv("Question_1\glass.csv")

# Preprocess the data
scaler = StandardScaler()
# X = scaler.fit_transform(glass_df.drop(columns=["Type"]))
X = glass_df.drop(columns=["Type"])
y = glass_df["Type"].values

# Define LOF Fuction
def lof(X, k):
    n = X.shape[0]
    lrd = np.zeros(n)
    lof = np.zeros(n)

    # Finding K-nearest Neighbors using mahalanobis distance as the metric
    nbrs = NearestNeighbors(
        n_neighbors=k,
        metric="mahalanobis",
        metric_params={"V": np.cov(X, rowvar=False)},
    ).fit(X)
    knnres = nbrs.kneighbors(X)
    distances, indices = knnres

    # Calculate reachability distance
    reach_dist = np.zeros(n)
    for i in range(n):
        reach_dist[i] = np.max(distances[i])

    # Calculate local reachability density of each point
    for i in range(n):
        lrd[i] = 1 / (np.sum(reach_dist[indices[i]]) / k)

    # Calculate local outlier factor of each point
    for i in range(n):
        lof[i] = np.sum(lrd[indices[i]] / (lrd[i])) / k

    # Return LOF
    return lof


# Compute LOF
lofScores = lof(X, k=5)
glass_df["LOF Score"] = lofScores
# print(lofScores.shape)
print(glass_df)


# Calculating outliers using lOF
lofs = np.array(lofScores)
hist, bin_edges = np.histogram(lofs, bins="auto")
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
thresh = bin_centers[np.argmax(hist)]
print("THRESHOLD LOF METHOD:", thresh)
outliner_indices = np.where(lofs >= thresh)[0]
inliner_indices = np.where(lofs < thresh)[0]

fig, ax = plt.subplots()
ax.scatter(
    glass_df.index, lofs, c=["black" if lof >= thresh else "red" for lof in lofs]
)
ax.set_xlabel("Indicies")
ax.set_ylabel("LOF Scores")

plt.show()

print(outliner_indices.shape)
