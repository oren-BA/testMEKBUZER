import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            sum = 0
            for k in range(X.shape[1]):
                sum += X[i][k]*X[k][j]
            C[i][j] = sum


@njit
def matmul_transpose_numba(X):
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in prange(X.shape[0]):
        for j in prange(X.shape[0]):
            sum = 0
            for k in prange(X.shape[1]):
                sum += X[i][k] * X[k][j]
            C[i][j] = sum


def matmul_transpose_gpu(X):
    X_gpu = cuda.to_device(X)
    C = np.zeros((X.shape[0], X.shape[0]))
    matmul_kernel[1, 1024](X_gpu, C)
    return C

@cuda.jit
def matmul_kernel(A, C):
    tx = cuda.threadIdx.x
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            s_arr = cuda.shared.array(1, dtype=np.int32)  # todo: I'm not sure regarding int64
            # todo: maybe we should initialized to 0
            iter_num = A.shape[1]//1024 if(A.shape[1] % 1024 ==0) else (A.shape[1]//1024 + 1)
            for k in range(iter_num):
                if 1024*k + tx < A.shape[1]:
                  cuda.atomic.add(s_arr, 0, A[i][k*1024 +tx]*A[k*1024 +tx][j])
            cuda.syncthreads()  # wait until all blocks finished

            if tx == 0:
                C[i][j] = s_arr[0]

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


if __name__ == '_main_':
    matmul_comparison()