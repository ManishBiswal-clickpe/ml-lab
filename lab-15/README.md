# Lab 15: K-Means clustering

Uses the emailed Liss_III.csv, selecting Band_2 and Band_3 (column positions 2 and 3), with two K-Means clusters, k-means++ initialization and random_state=42. Based on lab15.py attached to "ML Lab 14 and 15 code".

Run `python lab.py` from this folder after installing the repository requirements. Actual run: 400 samples, confusion matrix `[[196, 3], [6, 195]]`, reported OA 97.75%. Plot: `figures/kmeans_clusters.png`. Full console output is recorded in comments at the bottom of lab.py.

Preserves the supplied label comparison (`cluster_id + 1`). Cluster IDs are arbitrary, so this OA depends on label alignment; it is not supervised held-out classification accuracy. No scaling or relabeling was added. Replaced pre-recorded comments with actual execution output.
