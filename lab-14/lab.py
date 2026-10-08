from pathlib import Path
import matplotlib

matplotlib.use("Agg")
Path("figures").mkdir(exist_ok=True)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data
y = iris.target

print("Original Feature Shape:", X.shape)
print("Original Features:")
print(iris.feature_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nReduced Feature Shape:", X_pca.shape)

print("\nFirst 5 samples after PCA:")
print(X_pca[:5])

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Retained:")
print(sum(pca.explained_variance_ratio_))

plt.figure(figsize=(8, 6))

for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Feature Reduction - Iris Dataset")
plt.legend()
plt.grid(True)
plt.savefig("figures/pca_iris.png", dpi=150, bbox_inches="tight")
plt.close()

# Output from execution with repository requirements:
# Original Feature Shape: (150, 4)
# Original Features:
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#
# Reduced Feature Shape: (150, 2)
#
# First 5 samples after PCA:
# [[-2.26470281  0.4800266 ]
#  [-2.08096115 -0.67413356]
#  [-2.36422905 -0.34190802]
#  [-2.29938422 -0.59739451]
#  [-2.38984217  0.64683538]]
#
# Explained Variance Ratio:
# [0.72962445 0.22850762]
#
# Total Variance Retained:
# 0.9581320720000166

