
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay
iris=load_iris();X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=.25,random_state=42,stratify=iris.target)
model=DecisionTreeClassifier(max_depth=3,random_state=42);model.fit(X_train,y_train);pred=model.predict(X_test)
print('Test accuracy:',accuracy_score(y_test,pred));print(classification_report(y_test,pred,target_names=iris.target_names,zero_division=0))
new_sample=np.array([[5.1,3.5,1.4,0.2]])
print('New sample:',new_sample);print('Predicted class:',iris.target_names[model.predict(new_sample)[0]])
plt.figure(figsize=(14,8));plot_tree(model,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True);plt.title('Iris decision tree');plt.show()
ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=iris.target_names);plt.title('Decision tree test results');plt.show()
