
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('IRIS.csv', header=None)

dataset = dataset.values
X = dataset[:,0:4]
Y = dataset[:,4]
print(X)
print(Y)



plt.figure();plt.scatter(dataset[:,0],dataset[:,2],c=Y,cmap='viridis');plt.xlabel('Sepal length');plt.ylabel('Petal length');plt.title('Iris feature selection');plt.colorbar(label='Species code');plt.show()