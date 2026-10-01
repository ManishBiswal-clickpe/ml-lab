# Lab 11: K-nearest neighbors classifier

KNN on Age and EstimatedSalary to predict Purchased, using 11 neighbors, Minkowski distance with p=2, and the starter's 75/25 split with `random_state=0`. Prints predictions, confusion matrix, and overall accuracy, and saves training/test decision-boundary plots.

Classroom did not attach Social_Network_Ads.csv. This includes the standard public 400-row copy from [Machine-Learning-A-Z](https://github.com/atse0612/Machine-Learning-A-Z), retrieved from [this dataset blob](https://api.github.com/repos/atse0612/Machine-Learning-A-Z/git/blobs/4e6dadda64337a73070fa9d7b2196800cec259f0). Replace it with the course CSV if one is supplied later. It has User ID, Gender, Age, EstimatedSalary, and Purchased columns, matching the starter's column positions.

Preserves unscaled inputs rather than changing the posted model. Salary therefore dominates distances; the plots reflect that. Replaced the 0.01-step mesh with 250 x 250 grids so raw-salary plots fit memory. Uses `color=` for scatter colors and saves PNGs instead of opening plot windows. Converted the posted starter notebook to a script; no notebook submission was specified.

Run `python lab.py` from this folder after installing the repo's `requirements.txt`. PNG graphs are in `figures/`; printed output from a real run is in comments at the bottom of the script.
