from numpy import array
from numpy.linalg import qr

A = array([[1,2], [3,4], [5,6]])
print(A)

Q , R = qr(A, 'complete')
print(Q)
print(R)

B = Q.dot(R)
print(B)
