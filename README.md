# Cooperative-Groups
Tests for CUDA cooperative groups

`reduction.cu`: taken from the [cuda samples](https://github.com/NVIDIA/cuda-samples) and slightly modified, uses cooperative groups to perform a sum reduction.

`reduction_kernels.cu`: uses two kernel calls to perform the reduction.

`reduction_pseudo.cu`: uses a modified kernel to perform the reduction, with pseudo-grouping of threads.

## Compilation and execution on FinisTerrae III

`chmod +x compile_exec.sh && ./compile_exec.sh SIZE ITERS` where `SIZE` is the group size and ITERS is the number of iterations the program does. Their default values are 16 and 1000, respectively.
