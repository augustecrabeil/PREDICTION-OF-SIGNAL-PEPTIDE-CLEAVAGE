import sys
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.svm
import sklearn.datasets
import csv


# the BLOSUM62 matrix
M = np.array([[4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1,  1, 0, -3, -2, 0, -2, -1, 0, -4], 
            [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4], 
            [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4], 
            [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4], 
            [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4], 
            [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4], 
            [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4], 
            [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4], 
            [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4], 
            [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4], 
            [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4], 
            [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4], 
            [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4], 
            [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4], 
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4], 
            [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4], 
            [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4], 
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4], 
            [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4], 
            [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4], 
            [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4], 
            [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4], 
            [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4], 
            [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1]])



# the calculations for the unconventional kernels
def pseudo_dot(x1, x2) :
    sum = 0.0
    a1 = -1
    a2 = -1
    for i in range(len(x1)//26) :
        for j in range(26) :
            if (x1[(26*i)+j] == 1) :
                a1 = chr(ord('A') + j)
            if (x2[(26*i)+j] == 1) :
                a2 = chr(ord('A') + j)
        if (a1 == 'A') :
            a1 = 0
        if (a1 == 'R'):
            a1 = 1
        if (a1 == 'N'):
            a1 = 2
        if (a1 == 'D'):
            a1 = 3
        if (a1 == 'C'):
            a1 = 4
        if (a1 == 'Q'):
            a1 = 5
        if (a1 == 'E'):
            a1 = 6
        if (a1 == 'G'):
            a1 = 7
        if (a1 == 'H'):
            a1 = 8
        if (a1 == 'I'):
            a1 = 9
        if (a1 == 'L'):
            a1 = 10
        if (a1 == 'K'):
            a1 = 11
        if (a1 == 'M'):
            a1 = 12
        if (a1 == 'F'):
            a1 = 13
        if (a1 == 'P'):
            a1 = 14
        if (a1 == 'S'):
            a1 = 15
        if (a1 == 'T'):
            a1 = 16
        if (a1 == 'W'):
            a1 = 17
        if (a1 == 'Y'):
            a1 = 18
        if (a1 == 'V'):
            a1 = 19
        if (a2 == 'A'):
            a2 = 0
        if (a2 == 'R'):
            a2 = 1
        if (a2 == 'N'):
            a2 = 2
        if (a2 == 'D'):
            a2 = 3
        if (a2 == 'C'):
            a2 = 4
        if (a2 == 'Q'):
            a2 = 5
        if (a2 == 'E'):
            a2 = 6
        if (a2 == 'G'):
            a2 = 7
        if (a2 == 'H'):
            a2 = 8
        if (a2 == 'I'):
            a2 = 9
        if (a2 == 'L'):
            a2 = 10
        if (a2 == 'K'):
            a2 = 11
        if (a2 == 'M'):
            a2 = 12
        if (a2 == 'F'):
            a2 = 13
        if (a2 == 'P'):
            a2 = 14
        if (a2 == 'S'):
            a2 = 15
        if (a2 == 'T'):
            a2 = 16
        if (a2 == 'W'):
            a2 = 17
        if (a2 == 'Y'):
            a2 = 18
        if (a2 == 'V'):
            a2 = 19
        if ((a1 != -1) and (a2 != -1)) :
            sum += M[a1][a2]
    return sum

def ratquad(X, Y) :
    X = np.array(X)
    Y = np.array(Y)
    M = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
    M += coef0
    M = coef0 / M
    return M

def pseudo_linear(X, Y) :
    X = np.array(X)
    Y = np.array(Y)
    res = np.array([pseudo_dot(x, y) for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return res

def pseudo_poly(X, Y) :
    X = np.array(X)
    Y = np.array(Y)
    res = np.array([pow(((gamma * pseudo_dot(x, y)) + coef0), degree) for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return res

def pseudo_rbf(X, Y) :
    X = np.array(X)
    Y = np.array(Y)
    res = np.array([np.exp((-gamma) * pseudo_dot(x-y, x-y)) for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return res

def pseudo_sigmoid(X, Y) :
    X = np.array(X)
    Y = np.array(Y)
    res = np.array([np.tanh((gamma * pseudo_dot(x, y)) + coef0) for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return res

def pseudo_ratquad(X, Y) :
    X = np.array(X)
    Y = np.array(Y)
    res = np.array([coef0 / (pseudo_dot(x-y, x-y) + coef0) for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return res

def probabilistic_kernel(X, Y) :
    res = np.zeros((len(X), len(Y)))
    s = pd.read_csv("s.csv", sep = ' ',
                    index_col=0,
                    header=None)  # no name of columns
    s = np.array(s)
    for x in range(len(X)) :
        for y in range(len(Y)) :
            sum = 0.0
            x1 = np.array(X.iloc[x])
            x2 = np.array(Y.iloc[y])
            for i in range(len(x1)//26) :
                for j in range(26) :
                    if (x1[(26*i)+j] == 1) :
                        a1 = j
                    if (x2[(26*i)+j] == 1) :
                        a2 = j
                if (a1==a2) :
                    sum += s[a1][i] + np.log(1 + np.exp(s[a1][i]))
                else :
                    sum += s[a1][i] + s[a2][i]
            res[x][y] = np.exp(sum)
    return res

def custom_product_kernel(X, Y) :
    return probabilistic_kernel(X, Y) * pseudo_rbf(X, Y)

def custom_sum_kernel(X, Y) :
    return (0.5*probabilistic_kernel(X, Y)) + (0.5*pseudo_rbf(X, Y))

if __name__=="__main__":
    # we collect the information
    if len(sys.argv) == 10:
        readfrom_train = sys.argv[1]
        readfrom_test = sys.argv[2]
        kernel = sys.argv[3]
        degree = int(sys.argv[4])
        gamma = float(sys.argv[5])
        coef0 = float(sys.argv[6])
        C = float(sys.argv[7])
        lr = float(sys.argv[8])
        scaled = sys.argv[9]

    else:
        print("Syntax: python %s <dataset_train> <dataset_test> [<kernel> <degree> <gamma> <coef0> <C> <lr> <scaled>]" % sys.argv[0])
        exit(0)

    label = 0
    char_sep = ';'

    dataset_train = pd.read_csv(readfrom_train,
                                sep=char_sep,
                                index_col=0,
                                header=None)  # no name of columns
    dataset_test = pd.read_csv(readfrom_test, sep=char_sep, index_col=0,
                                header=None)

    
    X_train = dataset_train.drop(dataset_train.columns[label], axis=1)
    X_test = dataset_test.drop(dataset_test.columns[label], axis=1)

    y_train = dataset_train.iloc[:, label]
    y_test = dataset_test.iloc[:, label]

    if ((kernel == "linear") or (kernel == "poly") or (kernel == "rbf") or (kernel == "sigmoid")) :
        svm_model = sk.svm.SVC(C=C, kernel=kernel, tol=lr)  # to match libSVM's svm-train default parameters
    
    elif (kernel == "ratquad") :
        svm_model = sk.svm.SVC(C=C, kernel=ratquad, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters

    elif (kernel == "pseudo_linear") :
        svm_model = sk.svm.SVC(C=C, kernel=pseudo_linear, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters
    
    elif (kernel == "pseudo_poly") :
        svm_model = sk.svm.SVC(C=C, kernel=pseudo_poly, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters
    
    elif (kernel == "pseudo_rbf") :
        svm_model = sk.svm.SVC(C=C, kernel=pseudo_rbf, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters

    elif (kernel == "pseudo_sigmoid") :
        svm_model = sk.svm.SVC(C=C, kernel=pseudo_sigmoid, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters
    
    elif (kernel == "pseudo_ratquad") :
        svm_model = sk.svm.SVC(C=C, kernel=pseudo_ratquad, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters

    elif (kernel == "probabilistic") :
        svm_model = sk.svm.SVC(C=C, kernel=custom_sum_kernel, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters
        # svm_model = sk.svm.SVC(C=C, kernel=custom_product_kernel, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters
        # svm_model = sk.svm.SVC(C=C, kernel=probabilistic_kernel, degree=degree, gamma=gamma, coef0=coef0, tol=lr)  # to match libSVM's svm-train default parameters

    else :
        print("Wrong kernel")

    scaler = sk.preprocessing.MinMaxScaler(feature_range=(-1, 1))  # to match libSVM's svm-scale's default parameter
    svm_model.fit(X_train, y_train)


    if(scaled == "0") :
        predictions = svm_model.predict(X_test)
        with open('output.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(predictions)
        
        print("Accuracy:", np.sum(predictions == y_test) / X_test.shape[0],
            f"({np.sum(predictions == y_test)}/{X_test.shape[0]})")
            

    else :
        svm_model.fit(scaler.fit_transform(X_train), y_train)
        predictions = svm_model.predict(scaler.transform(X_test))
        with open('outputScaled.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(predictions)

        print("Accuracy after scaling:", np.sum(predictions == y_test) / X_test.shape[0],
            f"({np.sum(predictions == y_test)}/{X_test.shape[0]})")