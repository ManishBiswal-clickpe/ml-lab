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
