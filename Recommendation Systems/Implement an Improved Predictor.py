import numpy as np
from numpy.lib.arraysetops import unique
from scipy import sparse
import sklearn.metrics.pairwise as pd
import creat_A, RMSE

### test2
task2_test = np.genfromtxt('task---2.test', delimiter=',')
task2_train = np.genfromtxt('task---2.training', delimiter=',')

trainA, b = creat_A(task2_train, task2_train, need_b = True)
train2_output, train2_pred = RMSE(task2_train, trainA, b)
print("Task 2 val RMSE:", train2_output)

testA, _ = creat_A(task2_train, task2_test, need_b = False)
test2_output, test2_pred = RMSE(task2_test, testA, b)
print(f'Task 2 RMSE: {test2_output}')


## Create the Deviation matrix
def Deviation_matrix(train_data, train_pred):
    true_lebal = train_data[:,2]
    train_U = unique(train_data[:,0]).shape[0]
    train_M = unique(train_data[:,1]).shape[0]
    data_r = true_lebal - train_pred
    row =  train_data[:,0]-1
    col = train_data[:,1]-1
    D = sparse.csr_matrix((data_r, (row, col)), shape = (train_U, train_M))
    return D

deviation_matrix = Deviation_matrix(task2_train, train2_pred)
cosine_sim = pd.cosine_similarity(deviation_matrix.T)



def movie_correction_vector(train_data, cosine_sim, deviation_matrix):
    U = train_data[:,0].reshape(-1).astype(int)
    M = train_data[:,1].reshape(-1).astype(int)
    C = U.shape[0]
    correction_vector = np.zeros((C, 1))

    for i, (u, m) in enumerate(zip(U, M)):
        delta_m = cosine_sim[:, m-1]
        delta_m[m-1] = 0
        dev_u = deviation_matrix[u-1, :].toarray()
        correction_vector[i] = np.dot(dev_u, delta_m) \
                             / np.sum(np.abs(delta_m))

    return correction_vector.reshape(-1)

def my_cosine_similarity(deviation_matrix):
    N = deviation_matrix.shape[1]
    m1r = []
    m2r = []
    Delta = np.zeros( (N, N) )
    for i in range(N):
        for j in range(i):

            m1r, _ = deviation_matrix[:, i].nonzero()
            m2r, _ = deviation_matrix[:, j].nonzero()   
            idx = np.intersect1d(m1r, m2r)
            deviation_matrix = deviation_matrix.toarray()
            Delta[i, j] = deviation_matrix[idx, i].dot(deviation_matrix[idx, j]) \
                / (np.linalg.norm(deviation_matrix[idx,i])*np.linalg.norm(deviation_matrix[idx,j])) 
            
            deviation_matrix = sparse.csc_matrix(deviation_matrix)

    return Delta
correction_v = movie_correction_vector(task2_test, cosine_sim, deviation_matrix)

test2_pred += correction_v
C = task2_test.shape[0]
diff = task2_test[:,2] - test2_pred
RMSE = (np.sum((diff)**2)/ C)**0.5
print("Task 2 RMSE after considering similarity:", RMSE)
print("####### Testing done #########")


