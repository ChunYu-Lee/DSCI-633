import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from pdb import set_trace

def pca(X, n_components = 5):
    #  Use svd to perform PCA on X
    #  Inputs:
    #     X: input matrix
    #     n_components: number of principal components to keep
    #  Output:
    #     principal_components: the top n_components principal_components
    #     X_pca = X.dot(principal_components)

    U, s, Vh = svd(X)
    V = np.transpose(Vh)
    principal_components = V[:,:n_components]

    return principal_components

def vector_norm(x, norm="Min-Max"):
    # Calculate the normalized vector

    if norm == "Min-Max":
        x_norm = [ (x[i]-min(x))/(max(x)-min(x)) for i in range(len(x)) ]
    elif norm == "L1":
        abs_sum = np.sum(abs(x))
        x_norm = [ x[i]/abs_sum for i in range(len(x)) ]
    elif norm == "L2":
        sum_square = ((np.sum(x**2))**(1/2))
        x_norm = [ x[i]/sum_square for i in range(len(x)) ]
    elif norm == "Standard_Score":
        standard_deviation = np.std(x)
        average = np.avg(x)
        x_norm = [ (x[i]-average)/standard_deviation for i in range(len(x)) ]
    else:
        raise Exception("Unknown normlization.")
    return x_norm

def normalize(X, norm="Min-Max", axis = 1):
    #  Inputs:
    #     X: input matrix
    #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
    #     axis = 0: normalize rows
    #     axis = 1: normalize columns
    #  Output:
    #     X_norm: normalized matrix (numpy.array)

    X_norm = deepcopy(np.asarray(X))
    m, n = X_norm.shape
    if axis == 1:
        for col in range(n):
            X_norm[:,col] = vector_norm(X_norm[:,col], norm=norm)
    elif axis == 0:
        X_norm = np.array([vector_norm(X_norm[i], norm=norm) for i in range(m)])
    else:
        raise Exception("Unknown axis.")
    return X_norm

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    u = np.unique(y_array)
    # Write your own code below
    sample = []
    for i in range(len(u)):
        temp_arr = np.where(y_array == u[i])[0]
        target_sample_amount = int(np.ceil(ratio * len(temp_arr)))
        sampled = np.random.choice(len(temp_arr), target_sample_amount, replace = replace)
        temp_sample = list(temp_arr[sampled])

        for j in range (len(temp_sample)):
        	sample.append(temp_sample[j])
    sample = np.asarray(sample)


    return sample.astype(int)
