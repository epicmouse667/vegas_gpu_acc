from numba import cuda
import math
import numpy as np

# x = np.arange(int(1e8)).reshape(int(1e4), -1)
# class Solution:
#     @cuda.jit# Set "nopython" mode for best performance, equivalent to @njit
#     def go_fast(self,a,res): # Function is compiled to machine code when called the first time
#         trace = 0.0
#         for i in range(a.shape[0]):   # Numba likes loops
#             trace += np.tanh(a[i, i]) # Numba likes NumPy functions
#         res[0] = trace  
#     # Numba likes NumPy broadcasting

# threadsperblock = (256,256 )
# blockspergrid_x = math.ceil(x.shape[0]/100 / threadsperblock[0])
# blockspergrid_y = math.ceil(x.shape[1] /100 / threadsperblock[1])
# x_device=cuda.to_device(x)
# print("move x to device")
# res_device=cuda.device_array(1)
# blockspergrid = (blockspergrid_x, blockspergrid_y)
# solution = Solution()
# solution.go_fast[threadsperblock,blockspergrid](x_device,res_device)
# res = res_device.copy_to_host()
# print(res)

import vegas
import math
@vegas.batchintegrand
def f(x):
    dx2 = 0
    for d in range(100000):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * 1013.2118364296088
integral_range = 100000*[[-1,1]]
# integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])
integ = vegas.Integrator(integral_range)
result = integ(f, nitn=10, neval=1000)
print(result.summary())
print('result = %s    Q = %.2f' % (result, result.Q))