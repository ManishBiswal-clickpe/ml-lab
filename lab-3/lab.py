from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
Path("figures").mkdir(exist_ok=True)


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv(
    "IRIS.csv",
    header=None,
    names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"],
)

print("First 5 observations:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)


print("\n--- Particular Column ---")


print(df["SepalLengthCm"].head())


print("\n--- Summary Statistics ---")

print(df.describe())


print("\n--- Mean ---")

numeric_columns = df.select_dtypes(include=np.number).columns.drop("Species")

mean_values = df[numeric_columns].mean()
print(mean_values)

print("\n--- Standard Deviation ---")

std_values = df[numeric_columns].std()
print(std_values)


print("\n--- Duplicate Values ---")

print("Duplicates before removal:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Duplicates after removal:", df.duplicated().sum())


print("\n--- Missing Values ---")

print(df.isnull().sum())


for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())


if "Species" in df.columns:
    df["Species"] = df["Species"].fillna(df["Species"].mode()[0])

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


print("\n--- Standardization ---")

X = df[numeric_columns]

standard_scaler = StandardScaler()

X_standardized = standard_scaler.fit_transform(X)

X_standardized = pd.DataFrame(X_standardized, columns=numeric_columns)

print(X_standardized.head())


print("\n--- Normalization ---")

min_max_scaler = MinMaxScaler()

X_normalized = min_max_scaler.fit_transform(X)

X_normalized = pd.DataFrame(X_normalized, columns=numeric_columns)

print(X_normalized.head())


print("\n--- Data Balancing ---")

min_count = df["Species"].value_counts().min()
df = pd.concat(
    [group.sample(n=min_count, random_state=42) for _, group in df.groupby("Species")],
    ignore_index=True,
)
print("Balanced class counts:")
print(df["Species"].value_counts())

print("\nClass proportions:")
print(df["Species"].value_counts(normalize=True))


print("\n--- Train Test Split ---")

X = df[numeric_columns]
Y = df["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=Y
)

print("X_train:")
print(X_train.head())

print("\nX_test:")
print(X_test.head())

print("\nY_train:")
print(Y_train.head())

print("\nY_test:")
print(Y_test.head())

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)


df["Species"].value_counts().sort_index().plot.bar(
    title="Class counts after duplicate removal"
)
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
print(
    "Class balancing check: counts shown above; equal-count random undersampling is applied."
)

# --- Output ---
# First 5 observations:
#    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  Species
# 0            5.1           3.5            1.4           0.2        1
# 1            4.9           3.0            1.4           0.2        1
# 2            4.7           3.2            1.3           0.2        1
# 3            4.6           3.1            1.5           0.2        1
# 4            5.0           3.6            1.4           0.2        1
#
# Shape:
# (150, 5)
#
# Columns:
# Index(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
#        'Species'],
#       dtype='object')
#
# --- Particular Column ---
# 0    5.1
# 1    4.9
# 2    4.7
# 3    4.6
# 4    5.0
# Name: SepalLengthCm, dtype: float64
#
# --- Summary Statistics ---
#        SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm     Species
# count     150.000000    150.000000     150.000000    150.000000  150.000000
# mean        5.843333      3.057333       3.758000      1.199333    2.000000
# std         0.828066      0.435866       1.765298      0.762238    0.819232
# min         4.300000      2.000000       1.000000      0.100000    1.000000
# 25%         5.100000      2.800000       1.600000      0.300000    1.000000
# 50%         5.800000      3.000000       4.350000      1.300000    2.000000
# 75%         6.400000      3.300000       5.100000      1.800000    3.000000
# max         7.900000      4.400000       6.900000      2.500000    3.000000
#
# --- Mean ---
# SepalLengthCm    5.843333
# SepalWidthCm     3.057333
# PetalLengthCm    3.758000
# PetalWidthCm     1.199333
# dtype: float64
#
# --- Standard Deviation ---
# SepalLengthCm    0.828066
# SepalWidthCm     0.435866
# PetalLengthCm    1.765298
# PetalWidthCm     0.762238
# dtype: float64
#
# --- Duplicate Values ---
# Duplicates before removal: 1
# Duplicates after removal: 0
#
# --- Missing Values ---
# SepalLengthCm    0
# SepalWidthCm     0
# PetalLengthCm    0
# PetalWidthCm     0
# Species          0
# dtype: int64
#
# Missing values after preprocessing:
# SepalLengthCm    0
# SepalWidthCm     0
# PetalLengthCm    0
# PetalWidthCm     0
# Species          0
# dtype: int64
#
# --- Standardization ---
#    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
# 0      -0.898033      1.012401      -1.333255     -1.308624
# 1      -1.139562     -0.137353      -1.333255     -1.308624
# 2      -1.381091      0.322549      -1.390014     -1.308624
# 3      -1.501855      0.092598      -1.276496     -1.308624
# 4      -1.018798      1.242352      -1.333255     -1.308624
#
# --- Normalization ---
#    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
# 0       0.222222      0.625000       0.067797      0.041667
# 1       0.166667      0.416667       0.067797      0.041667
# 2       0.111111      0.500000       0.050847      0.041667
# 3       0.083333      0.458333       0.084746      0.041667
# 4       0.194444      0.666667       0.067797      0.041667
#
# --- Data Balancing ---
# Balanced class counts:
# Species
# 1    49
# 2    49
# 3    49
# Name: count, dtype: int64
#
# Class proportions:
# Species
# 1    0.333333
# 2    0.333333
# 3    0.333333
# Name: proportion, dtype: float64
#
# --- Train Test Split ---
# X_train:
#      SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
# 85             5.0           2.3            3.3           1.0
# 78             5.7           2.8            4.5           1.3
# 101            6.7           3.0            5.2           2.3
# 87             6.1           2.8            4.7           1.2
# 127            7.2           3.0            5.8           1.6
#
# X_test:
#      SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
# 74             7.0           3.2            4.7           1.4
# 65             6.1           3.0            4.6           1.4
# 139            7.7           2.6            6.9           2.3
# 42             5.7           3.8            1.7           0.3
# 144            5.8           2.8            5.1           2.4
#
# Y_train:
# 85     2
# 78     2
# 101    3
# 87     2
# 127    3
# Name: Species, dtype: int64
#
# Y_test:
# 74     2
# 65     2
# 139    3
# 42     1
# 144    3
# Name: Species, dtype: int64
#
# Shapes:
# X_train: (117, 4)
# X_test : (30, 4)
# Y_train: (117,)
# Y_test : (30,)
# Class balancing check: counts shown above; equal-count random undersampling is applied.
