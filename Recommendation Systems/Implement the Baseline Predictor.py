import numpy as np
from numpy.lib.arraysetops import unique
from scipy import sparse
import matplotlib.pyplot as plt


train = np.genfromtxt('peich508.training', delimiter=',', skip_header = 0)
test = np.genfromtxt('peich508.test', delimiter=',', skip_header = 0)

val_train = np.genfromtxt('verification.training', delimiter=',', skip_header = 0)
val_test = np.genfromtxt('verification.test', delimiter=',', skip_header = 0)

## creat A matrix, b star
def creat_A(train_data, data, need_b = False):
    
    N = data[:,1].shape[0]
    train_U = unique(train_data[:,0]).shape[0]
    train_M = unique(train_data[:,1]).shape[0]

    col = np.append(data[:,0], train_U -1 + data[:,1])
    row = np.append(range(N), range(N))
    data_one = np.ones(2*N)

    A = sparse.csr_matrix((data_one, (row.reshape(-1), col.reshape(-1))), shape = (N, (train_U+train_M)))
    b_star = 1

    #need to calculate b vector or not
    if need_b:
        c = data[:,2] - np.mean(data[:,2])
        ATA = A.transpose().dot(A)
        regular_term = 0.0001 * np.identity(A.shape[1])
        inv = np.linalg.inv((ATA.toarray() + regular_term))
        b_star = inv.dot(A.transpose().toarray().dot(c))

    return A, b_star

def RMSE(test_data, A, b):
    true_label = test_data[:,2]
    avg_r = np.mean(true_label)
    pred = avg_r + A.dot(b)
    low = 0
    high = 0
    for i in range(len(pred)):
        if pred[i] < 1:
            low += 1
            pred[i] = 1
        if pred[i] > 5:
            high += 1
            pred[i] = 5
    rmse = (np.sum((true_label - pred)**2)/(true_label.shape[0]))**0.5
    return [rmse, pred]


## create train A, b
trainA, b = creat_A(train, train, need_b = True)
testA, _ = creat_A(train, test, False)

## val
val_trainA, val_b = creat_A(val_train, val_train, True)
val_testA, _ = creat_A(val_train, val_test)

## test prediction
train_output, train_pred = RMSE(train, trainA, b)
test_output, test_pred = RMSE(test, testA, b)
print("Task 1 val RMSE:", train_output)
print("Task 1 test RMSE:", test_output)

## plot the picture
abs_error = np.abs(np.round(test_pred)-test[:,2])
plt.hist(abs_error, bins = "auto")


