# Machine learning labs

Executed notebooks, Python programs, printed outputs, and graphs.

Labs 5 and 11 are deferred. The active experiments are 1-4, 6-10, and 12-20.

## Run

Use Python 3.10 or newer. Install `requirements.txt`, then open each notebook from its own lab folder, or run `python lab.py` from that folder. Lab 1 also includes `basic_exercises.py` and `basic_exercises.ipynb`.

## Results

All 19 notebooks and 19 Python programs ran successfully. Notebook graphs are in `figures/`; graphs from separate script runs are in `script-figures/`. Execution records are in the two JSON reports.

## Data and implementation notes

Posted starters were used for labs 1, 2, 3, 6, 7 and 8. The other active experiments were written from the syllabus. Starter adaptation notes are in each lab README. Executable copies are the corrected versions; raw starter documents are not included.

Classroom datasets: Iris (150 headerless rows), Liss-III (400 rows), and Salary_Data (30 rows). Built-in scikit-learn datasets are used where each lab names them. Generated student, cost, blob, review, and grid data are labeled as examples, not real records.

Labs 18 and 19 use the 1,797-row digits dataset. This is a manageable larger sample, not a production-scale dataset. Lab 17 uses 40 authored review examples and has low measured held-out accuracy; it is not a production sentiment model. Lab 20 measures goal success, not supervised label accuracy.

## Experiments

- [Lab 1: NumPy, pandas, statistics and plots](lab-1/lab.ipynb)
- [Lab 2: Read CSV and select features](lab-2/lab.ipynb)
- [Lab 3: Preprocessing and train-test split](lab-3/lab.ipynb)
- [Lab 4: Perceptron learning](lab-4/lab.ipynb)
- [Lab 6: Simple linear regression](lab-6/lab.ipynb)
- [Lab 7: Distance metrics](lab-7/lab.ipynb)
- [Lab 8: Logistic regression](lab-8/lab.ipynb)
- [Lab 9: Decision tree classification](lab-9/lab.ipynb)
- [Lab 10: Regression for cost estimation](lab-10/lab.ipynb)
- [Lab 12: Naive Bayes classification](lab-12/lab.ipynb)
- [Lab 13: SVM, ROC and AUC](lab-13/lab.ipynb)
- [Lab 14: Principal component feature reduction](lab-14/lab.ipynb)
- [Lab 15: K-means with varying K](lab-15/lab.ipynb)
- [Lab 16: Bagging and boosting comparison](lab-16/lab.ipynb)
- [Lab 17: Review preprocessing and sentiment analysis](lab-17/lab.ipynb)
- [Lab 18: PCA on a larger sample dataset](lab-18/lab.ipynb)
- [Lab 19: UMAP dimensionality reduction](lab-19/lab.ipynb)
- [Lab 20: Reinforcement learning with Q-learning](lab-20/lab.ipynb)
