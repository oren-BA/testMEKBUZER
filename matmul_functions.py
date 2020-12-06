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


    # this works really well with m>=1000 (actually if this 1001,2001 it will not be that good..
    #tx = cuda.threadIdx.x
    #for i in range(A.shape[0]):
    #    for j in range(A.shape[0]):
    #        s_arr = cuda.shared.array(1, dtype=np.int32)  # todo: I'm not sure regarding int64
    #        # todo: maybe we should initialized to 0
    #        iter_num = A.shape[1]//1024 if(A.shape[1] % 1024 ==0) else (A.shape[1]//1024 + 1)
    #        for k in range(iter_num):
    #            if 1024*k + tx < A.shape[1]:
    #              cuda.atomic.add(s_arr, 0, A[i][k*1024 +tx]*A[k*1024 +tx][j])
    #        cuda.syncthreads()  # wait until all blocks finished
    #        if tx == 0:
    #            C[i][j] = s_arr[0]

    # ***********************
    #tx = int(cuda.threadIdx.x)
    #thread_rows = A.shape[0] // 1024
    #first_row = 0
    #if tx < A.shape[0] % 1024:
    #    thread_rows += 1
    #    first_row = tx*thread_rows
    #if tx > A.shape[0] % 1024:
    #    first_row = thread_rows*tx + A.shape[0] % 1024
    #for i in range(first_row, first_row + thread_rows):
    #    for j in range(A.shape[0]):
    #        sum = 0
    #        for k in range(A.shape[1]):
    #           sum += A[i][k]*A[k][j]
    #      C[i][j] = sum

    #n, m = A.shape
    #stide = (n**2)//1024
    #tx = cuda.threadIdx.x
    #i = tx*stide
    #while i < (tx+1)*stide:
    #    sum = 0
    #    for k in range(m):
    #        sum += A[i//n][k]*A[k][i % n]
    #    C[i//n][i % n] = sum
    #    i += 1
    #if tx < stide:  # some threads need to work a bit harder :)
    #    sum = 0
    #    for k in range(m):
    #        sum += A[(n**2 - tx)// n][k] * A[k][(n**2 - tx) % n]
    #    C[(n**2 - tx) // n][(n**2 - tx) % n] = sum



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