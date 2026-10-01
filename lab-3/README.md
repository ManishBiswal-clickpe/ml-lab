# Lab 3: Preprocessing and train-test split

Fixed the filename, supplied column names for the headerless Iris CSV, and excluded numeric Species labels from input features to avoid target leakage. Preserves the provided duplicate removal, missing-value handling and stratified split, with equal-count random undersampling for class balancing. Scaling demonstrations are descriptive; no model is trained on them.

Run `python lab.py` from this folder. PNG graphs are in `figures/`; printed output is in comments at the bottom of the script.
