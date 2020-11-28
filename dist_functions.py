import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def dist_cpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    sum = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            sum += abs(A[i,j] - B[i,j])**p
    return sum**(1/p)

@njit(parallel=True)
def dist_numba(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    sum = 0
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            sum += abs(A[i, j] - B[i, j]) ** p
    return sum ** (1 / p)

def dist_gpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    C = np.zeros(1, dtype=np.int32)
    dist_kernel[1000,1000](A, B, p, C)
    # todo: perhaps we should get A,B into the gpu. we should check the performance of the server
    return C**p

@cuda.jit
def dist_kernel(A, B, p, C):
    # Thread id in a 1D block

    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    bx = cuda.blockIdx.x
    s_arr = cuda.shared.array(1, dtype=np.int32)  #todo: I'm not sure regarding int64
    cuda.atomic.add(s_arr,0,abs(A[tx][bx] - B[tx][bx])**p)
    cuda.syncthreads() # wait until all blocks finished
    if tx == 0:
        cuda.atomic.add(C,0,s_arr[0])
    # todo: how to finish the funciton??


   
# this is the comparison function - keep it as it is.
def dist_comparison():
    A = np.random.randint(0,256,(1000, 1000))
    B = np.random.randint(0,256,(1000, 1000))
    p = [1, 2]

    def timer(f, q):
        return min(timeit.Timer(lambda: f(A, B, q)).repeat(3, 20))

    for power in p:
        print('p=' + str(power))
        print('     [*] CPU:', timer(dist_cpu,power))
        print('     [*] Numba:', timer(dist_numba,power))
        print('     [*] CUDA:', timer(dist_gpu, power))

if __name__ == '__main__':
    dist_comparison()
