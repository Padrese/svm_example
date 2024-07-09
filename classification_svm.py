from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, KFold

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

mem = Memory("./mycache")

#Reading file in libsvm format
@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return sp.sparse.lil_matrix(data[0]).toarray(), np.array(data[1]) #Data parsing from Scipy's sparse matrix to NumPy array format

def cross_validation(estimator, degrees, X_tr, y_tr):

    # Hyperparameter tuning using grid search for each value of the polynomial degree with respect to C
    param_grid = {"degree": degrees}
    grid_cv = KFold(n_splits=10, shuffle=False)

    grid_estimator= GridSearchCV(estimator=estimator, param_grid=param_grid, cv=grid_cv,
    refit=True, n_jobs=-1, verbose=3, error_score="raise", return_train_score=True).fit(X_tr, y_tr)  # Fit on the new estimator

    best_params_ = grid_estimator.best_params_
    best_score_ = grid_estimator.best_score_

    print("Best params: ", best_params_)
    print("Best score: ", best_score_)

    results = grid_estimator.cv_results_
    mean_train_scores = results["mean_train_score"]
    mean_test_scores = results['mean_test_score']

    # Extract support vectors of the best estimator
    best_estimator = grid_estimator.best_estimator_
    support_vectors = best_estimator.support_vectors_
    print(support_vectors.shape)
    print("Number of support vectors:", len(support_vectors))
    print("Support Vectors:", support_vectors)

    #Now we extract the support vectors that are on the marginal hyperplanes y_{i}(w^{T}x_{i}+b) = 1
    support_indices = best_estimator.support_ #Extract the indices of the support vectors
    marg_hyp_support_vectors = []

    '''
    As w is not directly accessible because we use a polynomial kernel and thus considering the problem on a input space of higher dimension,
    we need to extract instead the dual values to build back w using the equality established by the KKT conditions
    '''
    alpha = best_estimator.dual_coef_.T
    print(alpha.shape)
    print(X_tr.shape)
    print(y_tr.shape)
    w = np.sum([alpha[i]*y_tr[i]*X_tr[i] for i in range(len(X_tr))])
    b = best_estimator.intercept_
    for idx in support_indices:
        if y_tr[idx](w@X_tr[idx]+b) == 1:
            marg_hyp_support_vectors.append(X_tr[idx])

    print("Number of support vectors that are on the marginal hyperplane y_{i}(w^{T}x_{i}+b) = 1 (non-outliers):", len(marg_hyp_support_vectors))

    return mean_train_scores, mean_test_scores

def plot_model_performances(C_values, degrees, mean_test_errors_array):
    plt.figure(figsize=(10,10))

    plt.xticks(C_values)
    plt.xlabel("Values for C")
    plt.ylabel("Average CV error")

    labels = [f"d={degrees[i]}" for i in range(4)]
    plt.title("SVC performance on the satimage dataset with cross-validation on kernel polynomial degree d")

    for i in range(len(labels)):
        plt.plot(C_values, mean_test_errors_array[i], label=labels[i])

    plt.legend()
    plt.savefig("overall_performances.png")

    plt.show()

def plot_best_model_performances(degrees, best_training_errors, best_test_errors):
    plt.figure(figsize=(10,10))

    plt.xticks(range(1,len(best_training_errors)+1))
    plt.xlabel("Values for d")
    plt.ylabel("Average CV error")

    labels = ["train", "test"]
    plt.title("Best SVC performance on the satimage dataset as a function of polynomial degree d")

    plt.plot(degrees, best_training_errors, label=labels[0])
    plt.plot(degrees, best_test_errors, label=labels[1])

    plt.legend()
    plt.savefig("best_model_performances.png")

    plt.show()

def main():

    '''
    We consider here the satimage dataset present on http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    This dataset contains 3104 training examples, 1331 validation examples and 2000 testing examples.
    Each example has 36 features. They fall under one of the 6 different classes.
    '''

    filename_tr = "satimage.scale.tr"
    filename_val = "satimage.scale.val"
    filename_test = "satimage.scale.t"

    X_data_tr, y_data_tr = get_data(filename_tr)
    X_data_val, y_data_val = get_data(filename_val)

    #Fusing training and validation sets together
    X_tr = np.concatenate((X_data_tr, X_data_val))
    y_tr = np.concatenate((y_data_tr, y_data_val))
    X_test, y_test = get_data(filename_test)

    #Feature normalization
    X_tr = X_tr / np.linalg.norm(X_tr, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    #Binary labels transformation. Here we consider the problem of separating class 6 from the others
    y_tr = np.where(y_tr == 6, 1, 0)
    y_test = np.where(y_test == 6, 1, 0)

    #Values of C
    C_values = [i*1e-1 for i in range(1,11)]
    degrees = [i for i in range(1,5)]

    #Polynomial degree values

    for C in C_values:
        #We create the model. Here, the parameters of interest are the polynomial degree and the values taken by C
        print("C =",C)
        estimator = SVC(C=C,kernel="poly")

        mean_train_scores, mean_test_scores = cross_validation(estimator=estimator, degrees=degrees, X_tr=X_tr, y_tr=y_tr)
        mean_train_errors = np.array([1 - mean_train_scores[i] for i in range(len(mean_test_scores))])
        mean_test_errors = np.array([1 - mean_test_scores[i] for i in range(len(mean_test_scores))])
        if C != C_values[0]:
            mean_train_errors_array = np.vstack((mean_train_errors_array, mean_train_errors))
            mean_test_errors_array = np.vstack((mean_test_errors_array, mean_test_errors))
        else: 
            mean_train_errors_array = mean_train_errors
            mean_test_errors_array = mean_test_errors

    #We want the evolution of values as a function of C for every polynomial degree d, so the convenient information is stored in the columns
    plot_model_performances(C_values=C_values, degrees=degrees, mean_test_errors_array=mean_test_errors_array.T)

    #The results obtained clearly display C=1 and d=4 as being the best parameters to minimize the model errors.
    #We plot the training and test errors 
    best_training_errors = mean_train_errors_array[-1]
    best_test_errors = mean_test_errors_array[-1]

    plot_best_model_performances(degrees=degrees, best_training_errors=best_training_errors, best_test_errors=best_test_errors)


if __name__ == "__main__":
    main()