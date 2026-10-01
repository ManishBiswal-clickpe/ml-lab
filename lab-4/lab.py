
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)


from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
iris=load_iris(); X=iris.data; y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model=make_pipeline(StandardScaler(),Perceptron(max_iter=1000,tol=1e-3,random_state=42))
model.fit(X_train,y_train); pred=model.predict(X_test)
print('Test accuracy:',accuracy_score(y_test,pred));print(classification_report(y_test,pred,target_names=iris.target_names,zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=iris.target_names);plt.title('Perceptron on Iris');plt.show()
