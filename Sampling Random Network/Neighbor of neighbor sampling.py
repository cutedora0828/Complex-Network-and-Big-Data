import numpy as np
import random

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

## task 3 Neighbor-of-neighbor sampling
# calculate expected value of nn sampling
e_temp = 0
for i, A_row in enumerate(A):
    for r, c in zip(*A_row.nonzero()): 
        e_temp += data_attr[i-1] / data_attr[c-1]
e = e_temp / N
print(f'random connection of random node: {e}')

five_average_Neighbor_of_neighbor = []
for i in range(5):
    np.random.seed()
    idx = []
    for j in range(s):
        first_select = random.randint(1,len(data_attr))
        kk = len(A.getrow(first_select).nonzero()[1])
        second_select = random.randint(0, kk-1)
        second_attr = data_attr[second_select]
        idx.append(second_attr)
    five_average_Neighbor_of_neighbor.append(np.mean(idx))

Neighbor_of_neighbor_sampling = np.mean(five_average_Neighbor_of_neighbor)
print("random connection of random node:", Neighbor_of_neighbor_sampling)


