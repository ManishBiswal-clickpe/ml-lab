from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average="macro", zero_division=0)

recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

print("NAIVE BAYES CLASSIFIER - IRIS DATASET")
print("-------------------------------------")

print("Total number of samples:", len(X))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAccuracy:  {:.2f}%".format(accuracy * 100))
print("Precision: {:.4f}".format(precision))
print("Recall:    {:.4f}".format(recall))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=iris.target_names, zero_division=0
    )
)

# Output
# NAIVE BAYES CLASSIFIER - IRIS DATASET
# -------------------------------------
# Total number of samples: 150
# Training samples: 120
# Testing samples: 30
#
# Accuracy:  96.67%
# Precision: 0.9697
# Recall:    0.9667
#
# Confusion Matrix:
# [[10  0  0]
#  [ 0  9  1]
#  [ 0  0 10]]
#
# Classification Report:
#               precision    recall  f1-score   support
#
#       setosa       1.00      1.00      1.00        10
#   versicolor       1.00      0.90      0.95        10
#    virginica       0.91      1.00      0.95        10
#
#     accuracy                           0.97        30
#    macro avg       0.97      0.97      0.97        30
# weighted avg       0.97      0.97      0.97        30
#
