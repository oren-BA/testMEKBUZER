import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            sum = 0
            for k in range(X.shape[1]):
                sum += X[i][k]*X[j][k]
            C[i][j] = sum
    return C


@njit
def matmul_transpose_numba(X):
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in prange(X.shape[0]):
        for j in prange(X.shape[0]):
            sum = 0
            for k in prange(X.shape[1]):
                sum += X[i][k] *X[j][k]
            C[i][j] = sum
    return C

def matmul_transpose_gpu(X):
    X_gpu = cuda.to_device(X)
    C = np.zeros((X.shape[0], X.shape[0]))
    matmul_kernel[1, 1024](X_gpu, C)
    return C

@cuda.jit
def matmul_kernel(A, C):
    n, m = A.shape
    tx = cuda.threadIdx.x
    i = 0
    while 1024*i < n**2:
        if (1024*i + tx) < n**2:
            i_index = (1024*i+tx)//n
            j_index = (1024*i+tx) % n
            sum = 0
            for k in range(m):
                sum += A[i_index][k]*A[j_index][k]
            C[i_index][j_index] = sum
        i += 1



#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()