# Intro
WIP. This section contains resources related to developing and benchmarking algorithms for matrix multiplication on CPU solutions. This section aims to provide granular details of efforts required to understand matric multiplication on CPU systems and will expand on material covered in the article linked below.

* [Benchmarking Perfomance with OpenBLAS](https://salykova.github.io/matmul-cpu) 

# Step 1) General Matrix Multiplcation (GEMM)
The first section of this article introduces matrix multiplication as a core operation in modern computing, especially neural networks (we also know it is crucial to physics based engineering caluclations). It mentions how standard matmul operations found in Numpy, PyTorch, etc rely on external BLAS (Basic Linear Algebra Subprograms) libraries. These libraries contain optimized implementations of common linear algebra operations such as the dot product, matrix multiplication, vector addition, and scalar multplication. This article aims to develop and benchmarck a custom GEMM algorithm versus the *[OpenBLAS](https://en.wikipedia.org/wiki/OpenBLAS) routines. As an introduction, this section links several resources to familiarilze yourself with matmul algorithms on the CPU. We will summarize the contents of each and include developed solutions. 

## Fast Multidimensional Matrix Multiplication on CPU from Scratch
* [Link to Article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)

## Matrix Multiplication
* [Link to Article](https://en.algorithmica.org/hpc/algorithms/matmul/)

## Can you multiply a matrix? (YT Stream by George Hotz)
* [Link to Stream](https://www.youtube.com/watch?v=VgSQ1GOC86s)

# Step 2) How to Install and Benchmark OpenBLASS

# Step 3) Theoretical Limit

# Step 4) Naive Implementation

# Step 5) Kernel

# Step 6) Padding

# Step 7) Cach Blocking

# Step 8) Kernel Micro-Optimizations

# Step 9) Multithreading

