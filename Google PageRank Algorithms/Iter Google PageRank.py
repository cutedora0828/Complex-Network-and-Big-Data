import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


link1 = np.genfromtxt('links/1.txt', delimiter='')

f = open('titles/1.txt')
title1 = []
for line in f:
    title1.append(line)

link2 = np.genfromtxt('links/2.txt', delimiter='')

f = open('titles/2.txt')
title2 = []
for line in f:
    title2.append(line)


###### parameters
link = link1
title = title1

#####the process of cal Adj matrix
N = np.max(link).astype(int)
row = (link[:,1]-1).reshape(-1)
col = (link[:,0]-1).reshape(-1)
data = np.ones([len(link)]).reshape(-1)
A = csr_matrix((data, (row, col)), shape=(N, N))

u = np.ones([N])
in_degree = A.dot(u).reshape(-1)
out_degree = A.transpose().dot(u).reshape(-1)
normal_in_degree = list(in_degree/np.sum(in_degree))
del normal_in_degree[0]
normal_out_degree = list(out_degree/np.sum(out_degree))
del normal_out_degree[0]

AA = A.toarray().astype(float)
divide_b = np.tile(out_degree, (N,1))
H =  np.divide(AA, divide_b, out = np.full_like(AA, 1/(N)), where = divide_b != 0) 


alpha = 0.85
temp = np.linalg.inv(np.eye(N) - alpha * H )
PR = (1/N * temp.dot(np.ones([N, 1]))).reshape(-1) 
PR = list(PR/np.sum(PR))

dataf = {
"website": title,
"PR 0.85": PR
}
df_85 = pd.DataFrame(dataf)


