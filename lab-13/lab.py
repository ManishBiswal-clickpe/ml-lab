from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("SUPPORT VECTOR MACHINE - IRIS DATASET")
print("--------------------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy: {:.2f}%".format(accuracy * 100))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_prob = model.predict_proba(X_test)

auc_score = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")

print("Macro-average AUC: {:.4f}".format(auc_score))

class_names = iris.target_names

plt.figure(figsize=(8, 6))

for i in range(3):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_prob[:, i])

    class_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label="{} (AUC = {:.2f})".format(class_names[i], class_auc))

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - SVM on Iris Dataset")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(Path(__file__).with_name("roc.png"), dpi=150)
plt.close()

# Output
# SUPPORT VECTOR MACHINE - IRIS DATASET
# --------------------------------------
# Training samples: 120
# Testing samples: 30
# Accuracy: 96.67%
# Macro-average AUC: 0.9967
