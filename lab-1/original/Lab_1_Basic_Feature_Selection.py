




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