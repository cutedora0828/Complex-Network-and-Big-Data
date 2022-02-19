## task 5 Metropolis-Hastings random walk sampling
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

five_average_M_H_random_walk = []
for i in range(5):
    np.random.seed()
    first_select = random.randint(1, N)
    idx = []
    for j in range(s):
        kk = A.getrow(first_select).nonzero()[1]
        second_select = random.choices(kk)[0]
        
        p = data_attr[first_select-1] / data_attr[second_select-1]
        if p > 1:
            p=1
        if np.random.binomial(1, p, size=None) == 0 :
            second_select = first_select
        
        second_attr = data_attr[second_select-1]
        idx.append(second_attr)
        first_select = second_select

    five_average_M_H_random_walk.append(np.mean(idx))
M_H_random_walk = np.mean(five_average_M_H_random_walk)
print("M-H random walk:", M_H_random_walk)

