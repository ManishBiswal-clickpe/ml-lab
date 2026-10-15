# Lab 13: Support vector machine

Based on Lab13_SupportVectorMachine.txt in the [Classroom assignment](https://classroom.google.com/c/ODcxMzk3MDI5NDkz/a/ODkyMTY1MzQxMzY5/details?email=b523035@iiit-bh.ac.in&pli=1), posted October 14, 2026.

Runs an RBF SVM on scikit-learn's Iris dataset with a stratified 80/20 split and random_state=42. Fits StandardScaler only on the training data, reports accuracy and macro one-vs-rest AUC, and plots each class's ROC curve.

Run `python lab.py` after installing the repository requirements. Verified accuracy: 96.67%; macro AUC: 0.9967. Actual output is included at the bottom of lab.py; the graph is roc.png. The starter's explanatory comments and stray closing Markdown fence were removed. Replaced interactive display with headless PNG saving; the algorithm is unchanged.
