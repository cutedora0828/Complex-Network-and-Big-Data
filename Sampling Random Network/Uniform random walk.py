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

## task 4 Uniform random-walk
# calculate expected value of random walk sampling
e = (1 /N) * np.dot(data_attr, )
print(f'Expected value of random-walk sampling: {e}')
five_average_Uniform_random_walk = []
for i in range(5):
    np.random.seed()
    first_select = random.randint(1, N)
    idx = []
    for j in range(s):
        kk = A.getrow(first_select).nonzero()[1]
        second_select = random.choices(kk)[0]
        second_attr = data_attr[second_select-1]
        idx.append(second_attr)
        first_select = second_select
    five_average_Uniform_random_walk.append(np.mean(idx))
Uniform_random_walk_sampling = np.mean(five_average_Uniform_random_walk)
print("uniform random walk:", Uniform_random_walk_sampling)
