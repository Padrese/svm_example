# SVM example using scikit-learn

This project contains the implentation of a SVM classifier on the satimage dataset that we can find on http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.

This dataset contains 3104 training examples, 1331 validation examples and 2000 testing examples.
Each example has 36 features. They fall under one of the 6 different classes, and the goal here is to separate class 6 from the others using a classifier that maximizes the margin.

Cross-validation is used to find the best parameters d (degree of the polynomial kernel) and C (parameter linked to the xi variables representing the outliers in the non-separable case).
