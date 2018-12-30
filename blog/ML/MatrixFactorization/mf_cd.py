from numpy import array
from numpy.linalg import cholesky

A = array([[2, 1, 1,], [1, 2, 1], [1, 1, 2]])

print(A)

L =  cholesky(A)

print(L)

B = L.dot(L.T)
print(B)
