This package compute matrix-matrix multiplication using one-sided communication and Shared Memory collectives.
The code can be run as follows
mpirun -np 4 ./matmulRMA 4(matrix_size) 2(block size) 2(size of process in x dimesnion) 2(size of processor in y Dimesnion)

mpirun --host node01,node02,node03,node04 --bind-to none `pwd`/matmulRMA 4 2 2 2
