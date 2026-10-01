
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)


import numpy as np
a=np.array([10,20,30,40]); b=np.array([[1,2,3],[4,5,6]]); c=np.array([[[1,2],[3,4]],[[5,6],[7,8]]]); print(a); print(b); print(c)

arr=np.array([10,20,30,40,50]); print(arr[0]); print(arr[-1]); print(arr[1:4])

A=np.array([1,2,3]); B=np.array([4,5,6]); print(A+B); print(A-B); print(A*B); print(A/B)

data=np.array([12,15,18,20,25]); print(np.mean(data));print(np.median(data));print(np.std(data));print(np.max(data));print(np.min(data))

A=np.array([[1,2],[3,4]]); B=np.array([[5,6],[7,8]]);print(np.dot(A,B));print(A.T)

import pandas as pd
s=pd.Series([10,20,30,40]); print(s);df=pd.DataFrame({'Name':['Arun','Ravi','Priya'],'Age':[30,28,25],'Marks':[90,85,95]});print(df)

df=pd.read_csv('students.csv');print(df.head())

df=pd.DataFrame({'Name':['Arun','Ravi','Priya','Sita'],'Marks':[90,75,95,60]});print(df[df['Marks']>80])

df=pd.DataFrame({'Name':['Arun','Ravi','Priya'],'Marks':[90,np.nan,95]});df['Marks']=df['Marks'].fillna(df['Marks'].mean());print(df)

df=pd.DataFrame({'Marks':[90,80,85,95,88]});print(df.describe())

plt.figure();plt.plot([1,2,3,4,5],[2,4,6,8,10]);plt.title('Line Plot');plt.show()

plt.figure();plt.bar(['Python','C','Java','AI'],[85,70,90,95]);plt.title('Programming language marks');plt.show()

plt.figure();plt.hist([60,70,75,80,85,90,95,75,80],bins=5);plt.title('Marks histogram');plt.show()

plt.figure();plt.pie([30,20,15,35],labels=['Python','Java','C++','AI'],autopct='%1.1f%%');plt.title('Language shares');plt.show()

plt.figure();plt.scatter([150,155,160,165,170],[45,50,55,60,68]);plt.title('Height and weight');plt.xlabel('Height');plt.ylabel('Weight');plt.show()

df=pd.read_csv('students.csv');print(df.describe());plt.figure();plt.plot(df['Name'],df['Marks'],marker='o');plt.title('Example student marks');plt.show()

# --- Output ---
# [10 20 30 40]
# [[1 2 3]
#  [4 5 6]]
# [[[1 2]
#   [3 4]]
#
#  [[5 6]
#   [7 8]]]
# 10
# 50
# [20 30 40]
# [5 7 9]
# [-3 -3 -3]
# [ 4 10 18]
# [0.25 0.4  0.5 ]
# 18.0
# 18.0
# 4.427188724235731
# 25
# 12
# [[19 22]
#  [43 50]]
# [[1 3]
#  [2 4]]
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64
#     Name  Age  Marks
# 0   Arun   30     90
# 1   Ravi   28     85
# 2  Priya   25     95
#     Name  Age  Marks
# 0   Arun   30     90
# 1   Ravi   28     85
# 2  Priya   25     95
# 3   Sita   24     60
#     Name  Marks
# 0   Arun     90
# 2  Priya     95
#     Name  Marks
# 0   Arun   90.0
# 1   Ravi   92.5
# 2  Priya   95.0
#           Marks
# count   5.00000
# mean   87.60000
# std     5.59464
# min    80.00000
# 25%    85.00000
# 50%    88.00000
# 75%    90.00000
# max    95.00000
#              Age      Marks
# count   4.000000   4.000000
# mean   26.750000  82.500000
# std     2.753785  15.545632
# min    24.000000  60.000000
# 25%    24.750000  78.750000
# 50%    26.500000  87.500000
# 75%    28.500000  91.250000
# max    30.000000  95.000000
