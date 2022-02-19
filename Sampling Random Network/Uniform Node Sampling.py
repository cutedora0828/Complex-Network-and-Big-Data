import numpy as np
import random
from scipy import sparse

test_mod10star = np.loadtxt('mod10star.mat')
test_mod10star_attr = np.loadtxt('mod10star_attr.mat')
train_LiveJournal = np.loadtxt('LiveJournal.mat')
train_LiveJournal_attr = np.loadtxt('LiveJournal_attr.mat')


### parameters
s = 10000
data = train_LiveJournal
data_attr = train_LiveJournal_attr
print("Using LiveJournal")

data = test_mod10star
data_attr =test_mod10star_attr

N = int(np.max(data[:,0]))
col = data[:,0]
row = data[:,1]
data_one = np.ones(len(row))
A = sparse.csr_matrix((data_one, (row, col)), shape = (N+1, N+1))

print("-- expected values of <x>-hat -----")

# Task 1: Exact average
exact = np.sum(data_attr) / N
print(f'Expected value of uniform sampling: {exact}')

## task 2 Uniform node sampling
five_average_Uniform_node = []
for j in range(5):
    idx = []
    np.random.seed()
    for i in range(s):
        k = random.randint(0,len(data_attr)-1)
        idx.append(data_attr[k])
    five_average_Uniform_node.append(np.mean(idx))

Uniform_node_sampling = np.mean(five_average_Uniform_node)
print("uniform sampling:", Uniform_node_sampling)
