
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split





df = pd.read_csv("IRIS.csv", header=None, names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])

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

print("Duplicates before removal:",
      df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Duplicates after removal:",
      df.duplicated().sum())



print("\n--- Missing Values ---")

print(df.isnull().sum())



for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())


if "Species" in df.columns:
    df["Species"] = df["Species"].fillna(
        df["Species"].mode()[0]
    )

print("\nMissing values after preprocessing:")
print(df.isnull().sum())



print("\n--- Standardization ---")

X = df[numeric_columns]

standard_scaler = StandardScaler()

X_standardized = standard_scaler.fit_transform(X)

X_standardized = pd.DataFrame(
    X_standardized,
    columns=numeric_columns
)

print(X_standardized.head())



print("\n--- Normalization ---")

min_max_scaler = MinMaxScaler()

X_normalized = min_max_scaler.fit_transform(X)

X_normalized = pd.DataFrame(
    X_normalized,
    columns=numeric_columns
)

print(X_normalized.head())



print("\n--- Data Balancing ---")

print(df["Species"].value_counts())

print("\nClass proportions:")
print(df["Species"].value_counts(normalize=True))






print("\n--- Train Test Split ---")

X = df[numeric_columns]
Y = df["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
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


df['Species'].value_counts().sort_index().plot.bar(title='Class counts after duplicate removal');plt.xlabel('Species');plt.ylabel('Count');plt.tight_layout();plt.show()
print('Class balancing check: counts shown above; no synthetic resampling is applied.')