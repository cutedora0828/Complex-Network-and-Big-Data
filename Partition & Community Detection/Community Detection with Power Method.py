import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append("../common-functions")

filename = 'SB-large-network'
data_is_SB = True
### Read in .mat file (graph)
print('Reading graph file...')
link_n = 0
with open(filename + '.txt') as lf:
    lines = lf.readlines()
    for line in lines:
        link_n += 1

# count node num
node_n = int(np.unique(row)[-1]) + 1
print(f'Node num in graph: {node_n}')
print(f'Link num in graph: {link_n}')
A = csr_matrix((value, (row, col)), shape=(node_n, node_n))
print('done')

# construct a degree vector k for efficient access
print('building degree vector k ...')
k = np.zeros((node_n, 1))
for r in row:
    r = int(r)
    k[r] = k[r] + 1
print('done')

def power_method(A, k, link_n, lbda_neg=None, error = 0.00000001):

    M = link_n
    N = A.shape[0]
    np.random.seed()
    x = np.random.rand(N, 1)
    x /= np.linalg.norm(x)
    control = True

    while control:
        if lbda_neg:
            new_x = A.dot(x) - (np.dot(k.T, x) / (2*M)) * k - lbda_neg * x
        else:
            new_x = A.dot(x) - (np.dot(k.T, x) / (2*M)) * k 
        new_x /= np.linalg.norm(new_x)
        new_Zx = A.dot(new_x) - (np.dot(k.T, new_x) / (2*M))* k 
        landa = new_x.T.dot(new_Zx)[0, 0]
        e = np.linalg.norm(x - new_x) 
        if e < error or 2-e < error:
            control = False
        x = new_x
            
    return new_x, landa

x, lbda = power_method(A, k, link_n, error = 0.00000001)
print(lbda)

if lbda < 0:
    lbda_neg = lbda
    x, lbda = power_method(A, k, link_n, lbda_neg, error = 0.00000001)
    print(lbda)

### Community Detection
# Get all the indices in x which > 0 (belongs to group A)
x = [int(i > 0) for i in x]
iA = []
iB = []
for i, pos in enumerate(x):
    if pos:
        iA.append(i)
    else:
        iB.append(i)

# Generate the graph and color it with our community detection
G = snap.LoadEdgeList(snap.PUNGraph, filename + ".txt", 0, 1)
save_csrmatrix_Gephi_gexf_twocolors(A,filename + '.gexf',x)
print('done')
