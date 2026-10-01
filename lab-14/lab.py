
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
D=load_iris();scaled=StandardScaler().fit_transform(D.data);pca=PCA(n_components=2);embedded=pca.fit_transform(scaled)
print('Original shape:',D.data.shape);print('Reduced shape:',embedded.shape);print('Explained variance ratios:',pca.explained_variance_ratio_);print('Retained variance:',pca.explained_variance_ratio_.sum())
plt.figure();
for i,name in enumerate(D.target_names):
 plt.scatter(embedded[D.target==i,0],embedded[D.target==i,1],label=name)
plt.xlabel('PC1');plt.ylabel('PC2');plt.title('Iris PCA: 4 features to 2');plt.legend();plt.show()
