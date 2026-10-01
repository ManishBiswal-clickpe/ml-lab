

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('IRIS.csv')

dataset = dataset.values
X = dataset[:,0:4]
Y = dataset[:,4]
print(X)
print(Y)

