
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
Path('figures').mkdir(exist_ok=True)







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode





marks = [45, 67, 89, 56, 78, 91, 62, 74, 85, 69]

print("Dataset:")
print(marks)





mean = np.mean(marks)
print("\nMean =", mean)





median = np.median(marks)
print("Median =", median)





data = [2,4,5,5,6,7,5,8,9]

print("Mode =", mode(data))





print("\nMaximum =", max(marks))
print("Minimum =", min(marks))





Range = max(marks) - min(marks)
print("Range =", Range)





Q1 = np.percentile(marks,25)
Q2 = np.percentile(marks,50)
Q3 = np.percentile(marks,75)

print("\nQ1 =",Q1)
print("Q2 =",Q2)
print("Q3 =",Q3)





IQR = Q3 - Q1

print("IQR =",IQR)





print("\nPopulation Variance =",np.var(marks))
print("Sample Variance =",np.var(marks,ddof=1))





print("\nPopulation Standard Deviation =",np.std(marks))
print("Sample Standard Deviation =",np.std(marks,ddof=1))





mean = np.mean(marks)
std = np.std(marks)

z = (np.array(marks)-mean)/std

print("\nZ Scores")
print(np.round(z,2))





df = pd.DataFrame({'Marks':marks})

print("\nSummary Statistics")
print(df.describe())





plt.figure(figsize=(6,4))
plt.hist(marks,bins=5)
plt.title("Histogram")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()





plt.figure(figsize=(5,4))
plt.boxplot(marks)
plt.title("Box Plot")
plt.show()





subjects=["Math","Physics","Chemistry","English","Biology"]
values=[80,75,90,85,78]

plt.figure(figsize=(6,4))
plt.bar(subjects,values)
plt.title("Bar Chart")
plt.ylabel("Marks")
plt.show()





plt.figure(figsize=(6,6))
plt.pie(values,
        labels=subjects,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Pie Chart")
plt.show()





days=[1,2,3,4,5,6,7]
temperature=[30,31,29,33,35,34,32]

plt.figure(figsize=(6,4))
plt.plot(days,temperature,marker='o')
plt.title("Line Graph")
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.grid(True)
plt.show()

print("\nProgram Completed Successfully.")

"""
Module 1: Statistics and Data Analysis
Google Colab / Python Examples
"""

import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from scipy.stats import gmean

def arithmetic_mean():
    data=[12,18,15,20,17,25,19]
    print("Arithmetic Mean:", np.mean(data))

def weighted_mean():
    marks=[70,80,90,95]
    credits=[3,4,2,5]
    print("Weighted Mean:", np.average(marks, weights=credits))

def median_demo():
    print("Median:", np.median([18,25,10,35,14,22,40]))

def mode_demo():
    print("Mode:", mode([2,3,4,4,5,5,5,6,6]))

def geometric_mean():
    print("Geometric Mean:", gmean([2,4,8,16]))

def dispersion():
    data=np.array([10,15,20,25,30])
    print("Range:", data.max()-data.min())
    print("Variance:", np.var(data))
    print("Std Dev:", np.std(data))
    print("CV:", np.std(data)/np.mean(data)*100)

def quartiles():
    data=[5,8,12,15,20,24,28,35]
    print("Q1:", np.percentile(data,25))
    print("Median:", np.percentile(data,50))
    print("Q3:", np.percentile(data,75))

def zscore():
    data=np.array([12,15,18,20,25])
    print((data-data.mean())/data.std())

def plots():
    plt.figure()
    plt.hist(np.random.normal(50,10,500), bins=20)
    plt.title("Histogram")
    plt.figure()
    plt.boxplot([12,14,15,17,20,22,24,25,30])
    plt.title("Box Plot")
    plt.figure()
    plt.bar(['Math','Physics','Chemistry','Biology'], [85,72,90,80])
    plt.title("Bar Chart")
    plt.figure()
    plt.pie([30,25,20,25], labels=['A','B','C','D'], autopct='%1.1f%%')
    plt.title("Pie Chart")
    plt.figure()
    plt.plot([1,2,3,4,5],[10,18,15,25,30], marker='o')
    plt.title("Line Graph")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    arithmetic_mean()
    weighted_mean()
    median_demo()
    mode_demo()
    geometric_mean()
    dispersion()
    quartiles()
    zscore()
    plots()



values = [45,67,89,56,78,91,62,74]
print('Range:', max(values)-min(values))
print('Q1, median, Q3:', np.percentile(values,[25,50,75]))
print('IQR:', np.percentile(values,75)-np.percentile(values,25))

# --- Output ---
# Dataset:
# [45, 67, 89, 56, 78, 91, 62, 74, 85, 69]
#
# Mean = 71.6
# Median = 71.5
# Mode = 5
#
# Maximum = 91
# Minimum = 45
# Range = 46
#
# Q1 = 63.25
# Q2 = 71.5
# Q3 = 83.25
# IQR = 20.0
#
# Population Variance = 197.64000000000001
# Sample Variance = 219.60000000000002
#
# Population Standard Deviation = 14.058449416631978
# Sample Standard Deviation = 14.818906842274163
#
# Z Scores
# [-1.89 -0.33  1.24 -1.11  0.46  1.38 -0.68  0.17  0.95 -0.18]
#
# Summary Statistics
#            Marks
# count  10.000000
# mean   71.600000
# std    14.818907
# min    45.000000
# 25%    63.250000
# 50%    71.500000
# 75%    83.250000
# max    91.000000
#
# Program Completed Successfully.
# Arithmetic Mean: 18.0
# Weighted Mean: 84.64285714285714
# Median: 22.0
# Mode: 5
# Geometric Mean: 5.656854249492379
# Range: 20
# Variance: 50.0
# Std Dev: 7.0710678118654755
# CV: 35.35533905932738
# Q1: 11.0
# Median: 17.5
# Q3: 25.0
# [-1.35526185 -0.67763093  0.          0.45175395  1.58113883]
# Range: 46
# Q1, median, Q3: [60.5  70.5  80.75]
# IQR: 20.25
