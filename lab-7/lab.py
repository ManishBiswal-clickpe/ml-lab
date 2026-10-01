from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
Path("figures").mkdir(exist_ok=True)


import builtins

_inputs = iter(["0", "50", "1", "2", "3", "3", "4", "5", "6", "7", "8"])


def recorded_input(prompt=""):
    value = next(_inputs)
    print(prompt + value)
    return value


input = recorded_input


import numpy as np
from sklearn.datasets import load_iris


def euclidean_distance(p, q):
    """
    Euclidean Distance:
    d(P,Q) = sqrt(sum((Pi - Qi)^2))
    """
    return np.sqrt(np.sum((p - q) ** 2))


def manhattan_distance(p, q):
    """
    Manhattan Distance:
    d(P,Q) = sum(|Pi - Qi|)
    """
    return np.sum(np.abs(p - q))


def minkowski_distance(p, q, r):
    """
    Minkowski Distance:
    d(P,Q) = (sum(|Pi - Qi|^r))^(1/r)
    """
    return np.sum(np.abs(p - q) ** r) ** (1 / r)


def chebyshev_distance(p, q):
    """
    Chebyshev Distance:
    d(P,Q) = max(|Pi - Qi|)
    """
    return np.max(np.abs(p - q))


def squared_euclidean_distance(p, q):
    """
    Squared Euclidean Distance:
    d(P,Q) = sum((Pi - Qi)^2)
    """
    return np.sum((p - q) ** 2)


def cosine_distance(p, q):
    """
    Cosine Distance:
    d(P,Q) = 1 - (P.Q)/(||P|| ||Q||)
    """
    dot_product = np.dot(p, q)
    magnitude_p = np.linalg.norm(p)
    magnitude_q = np.linalg.norm(q)

    if magnitude_p == 0 or magnitude_q == 0:
        return 1.0

    cosine_similarity = dot_product / (magnitude_p * magnitude_q)

    return 1 - cosine_similarity


iris = load_iris()

X = iris.data

feature_names = iris.feature_names


print("=" * 70)
print("       DISTANCE MEASURES IN N-DIMENSIONAL FEATURE SPACE")
print("=" * 70)

print("\nDataset: Iris Dataset")
print("Number of samples :", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFeatures:")
for i, name in enumerate(feature_names):
    print(f"{i + 1}. {name}")


print("\n" + "=" * 70)
print("FIRST 10 SAMPLE POINTS")
print("=" * 70)

print("\nSample\tSepal Length\tSepal Width\tPetal Length\tPetal Width")

for i in range(10):
    print(f"{i}\t{X[i][0]:.2f}\t\t{X[i][1]:.2f}\t\t{X[i][2]:.2f}\t\t{X[i][3]:.2f}")


print("\n" + "=" * 70)
print("SELECT TWO SAMPLE POINTS")
print("=" * 70)

point1_index = int(input("\nEnter index of first point (0-149): "))
point2_index = int(input("Enter index of second point (0-149): "))


if point1_index < 0 or point1_index >= len(X):
    print("Invalid first point index.")
    exit()

if point2_index < 0 or point2_index >= len(X):
    print("Invalid second point index.")
    exit()


P = X[point1_index]
Q = X[point2_index]


print("\nPoint P:")
print(P)

print("\nPoint Q:")
print(Q)

print("\nDifference P - Q:")
print(P - Q)


while True:
    print("\n" + "=" * 70)
    print("DISTANCE MEASURES")
    print("=" * 70)

    print("1. Euclidean Distance")
    print("2. Manhattan Distance")
    print("3. Minkowski Distance")
    print("4. Chebyshev Distance")
    print("5. Squared Euclidean Distance")
    print("6. Cosine Distance")
    print("7. Calculate ALL distances")
    print("8. Exit")

    choice = int(input("\nEnter your choice: "))

    if choice == 1:
        d = euclidean_distance(P, Q)

        print("\nEuclidean Distance =", d)

    elif choice == 2:
        d = manhattan_distance(P, Q)

        print("\nManhattan Distance =", d)

    elif choice == 3:
        r = float(input("\nEnter Minkowski parameter r: "))

        if r <= 0:
            print("r must be greater than 0.")

        else:
            d = minkowski_distance(P, Q, r)

            print("\nMinkowski Distance =", d)

    elif choice == 4:
        d = chebyshev_distance(P, Q)

        print("\nChebyshev Distance =", d)

    elif choice == 5:
        d = squared_euclidean_distance(P, Q)

        print("\nSquared Euclidean Distance =", d)

    elif choice == 6:
        d = cosine_distance(P, Q)

        print("\nCosine Distance =", d)

    elif choice == 7:
        print("\n" + "=" * 70)
        print("DISTANCE RESULTS")
        print("=" * 70)

        print("Euclidean Distance        :", euclidean_distance(P, Q))

        print("Manhattan Distance        :", manhattan_distance(P, Q))

        print("Chebyshev Distance        :", chebyshev_distance(P, Q))

        print("Squared Euclidean Distance:", squared_euclidean_distance(P, Q))

        print("Cosine Distance            :", cosine_distance(P, Q))

        r = 3

        print(f"Minkowski Distance (r={r}):", minkowski_distance(P, Q, r))

    elif choice == 8:
        print("\nProgram terminated.")
        break

    else:
        print("\nInvalid choice. Please select 1-8.")

names = [
    "Euclidean",
    "Manhattan",
    "Minkowski r=3",
    "Chebyshev",
    "Squared Euclidean",
    "Cosine",
]
values = [
    euclidean_distance(P, Q),
    manhattan_distance(P, Q),
    minkowski_distance(P, Q, 3),
    chebyshev_distance(P, Q),
    squared_euclidean_distance(P, Q),
    cosine_distance(P, Q),
]
import pandas as pd

print(pd.DataFrame({"Metric": names, "Distance": values}).to_string(index=False))
plt.figure(figsize=(9, 4))
plt.bar(names, values)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Distance (metric-specific units)")
plt.title("Iris points 0 and 50")
plt.tight_layout()
plt.show()

# --- Output ---
# ======================================================================
#        DISTANCE MEASURES IN N-DIMENSIONAL FEATURE SPACE
# ======================================================================
#
# Dataset: Iris Dataset
# Number of samples : 150
# Number of features: 4
#
# Features:
# 1. sepal length (cm)
# 2. sepal width (cm)
# 3. petal length (cm)
# 4. petal width (cm)
#
# ======================================================================
# FIRST 10 SAMPLE POINTS
# ======================================================================
#
# Sample	Sepal Length	Sepal Width	Petal Length	Petal Width
# 0	5.10		3.50		1.40		0.20
# 1	4.90		3.00		1.40		0.20
# 2	4.70		3.20		1.30		0.20
# 3	4.60		3.10		1.50		0.20
# 4	5.00		3.60		1.40		0.20
# 5	5.40		3.90		1.70		0.40
# 6	4.60		3.40		1.40		0.30
# 7	5.00		3.40		1.50		0.20
# 8	4.40		2.90		1.40		0.20
# 9	4.90		3.10		1.50		0.10
#
# ======================================================================
# SELECT TWO SAMPLE POINTS
# ======================================================================
#
# Enter index of first point (0-149): 0
# Enter index of second point (0-149): 50
#
# Point P:
# [5.1 3.5 1.4 0.2]
#
# Point Q:
# [7.  3.2 4.7 1.4]
#
# Difference P - Q:
# [-1.9  0.3 -3.3 -1.2]
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 1
#
# Euclidean Distance = 4.003748243833521
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 2
#
# Manhattan Distance = 6.7
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 3
#
# Enter Minkowski parameter r: 3
#
# Minkowski Distance = 3.5450237756877807
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 4
#
# Chebyshev Distance = 3.3000000000000003
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 5
#
# Squared Euclidean Distance = 16.030000000000005
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 6
#
# Cosine Distance = 0.07161964128508791
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 7
#
# ======================================================================
# DISTANCE RESULTS
# ======================================================================
# Euclidean Distance        : 4.003748243833521
# Manhattan Distance        : 6.7
# Chebyshev Distance        : 3.3000000000000003
# Squared Euclidean Distance: 16.030000000000005
# Cosine Distance            : 0.07161964128508791
# Minkowski Distance (r=3): 3.5450237756877807
#
# ======================================================================
# DISTANCE MEASURES
# ======================================================================
# 1. Euclidean Distance
# 2. Manhattan Distance
# 3. Minkowski Distance
# 4. Chebyshev Distance
# 5. Squared Euclidean Distance
# 6. Cosine Distance
# 7. Calculate ALL distances
# 8. Exit
#
# Enter your choice: 8
#
# Program terminated.
#            Metric  Distance
#         Euclidean  4.003748
#         Manhattan  6.700000
#     Minkowski r=3  3.545024
#         Chebyshev  3.300000
# Squared Euclidean 16.030000
#            Cosine  0.071620
