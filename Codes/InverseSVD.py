from numpy import array
from numpy import diag
from numpy import dot
from scipy.linalg import svd

if __name__ == '__main__': 
    # define a matrix
    A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(A)
    # Singular-value decomposition
    U, s, VT = svd(A)
    # create n x n Sigma matrix
    Sigma = diag(s)
    # reconstruct matrix
    B = U.dot(Sigma.dot(VT))
    print(B)