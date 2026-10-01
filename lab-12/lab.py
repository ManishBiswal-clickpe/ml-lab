
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)


from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,ConfusionMatrixDisplay
iris=load_iris();X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=.25,random_state=42,stratify=iris.target)
model=GaussianNB();model.fit(X_train,y_train);pred=model.predict(X_test)
print('Accuracy:',accuracy_score(y_test,pred));print('Macro precision:',precision_score(y_test,pred,average='macro',zero_division=0));print('Macro recall:',recall_score(y_test,pred,average='macro',zero_division=0));print(classification_report(y_test,pred,target_names=iris.target_names))
print('First five test cases:');print(np.column_stack([y_test[:5],pred[:5]]))
ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=iris.target_names);plt.title('Gaussian Naive Bayes');plt.show()
