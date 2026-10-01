
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Liss_III.csv')
X = dataset.iloc[:, [1, 2]].values
Y = dataset.iloc[:, 5].values
print(X)
print(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


Y_pred = classifier.predict(X_test)
print(Y_pred)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

print(Y_test)
print(len(Y_test))

Error=Y_pred-Y_test
OA = (cm[0,0]+cm[1,1])/len(Y_test)*100
print('OA :',OA,'%')





from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.linspace(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 250),
                     np.linspace(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 250))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'darkorange')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                color = ListedColormap(('blue', 'darkorange'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Band 1')
plt.ylabel('Band 2')
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.linspace(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 250),
                     np.linspace(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 250))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'darkorange')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                color = ListedColormap(('blue', 'darkorange'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Band 1')
plt.ylabel('Band 2')
plt.legend()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay,classification_report
print(classification_report(Y_test,Y_pred,zero_division=0))
ConfusionMatrixDisplay.from_predictions(Y_test,Y_pred);plt.title('Logistic regression test confusion matrix');plt.show()