"""
    Fast Multidimensional Matrix Multiplication on CPU from Scratch
    Author: Simon Boehm
    Date: August 2022

    https://siboehm.com/articles/22/Fast-MMM-on-CPU

    This file contains a solution to the exercise in the linked article.

"""

import numpy as np
import time
import matplotlib.pyplot as plt

def MMM(A, B):
    """
    Matrix Multiplication of two 2D matrices A and B.
    A has shape (m, k) and B has shape (k, n).
    The result C has shape (m, n) and is computed as:
    C[i, j] = sum_{k} A[i, k] * B[k, j]
    """
    C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)

    for row in range(A.shape[0]):  # m
        for col in range(B.shape[1]):  # n
            C[row, col] = sum(A[row, k] * B[k, col] for k in range(A.shape[1]))

    return C

if __name__ == "__main__":

    # Create empty lists to store execution times
    execution_times_MMM = []
    execution_times_numpy = []

    matrix_sizes = []

    # Loop through different sizes of matrices
    n = 20  # Adjust this value to change maximum the size of the matrices
    for i in range(n):
        print(f"Running MMM for n = {i}...")
        # Define the dimensions of the tensors
        # For example, n = 2 -- > 2^2 = 4 will create 4x4 matrices
        # n = 8 # will create 256x256 matrices
        X, Y = 1 << i, 1 << i

        # Create random tensors A and B
        # A: (X, Y), B: (Y, X) for proper matrix multiplication A @ B -> (X, X)
        A = np.random.rand(X, Y).astype(np.float32)
        B = np.random.rand(Y, X).astype(np.float32)

        # Measure the execution time of the MMM function
        start_time_MMM = time.time()
        C_MMM = MMM(A, B)
        end_time_MMM = time.time()
        execution_times_MMM.append(end_time_MMM - start_time_MMM)
        print(f"MMM Execution Time: {end_time_MMM - start_time_MMM:.6f} seconds")
        print(f"Result Shape: {C_MMM.shape}")
        print(f"Result Sample: {C_MMM[0, :5]}")  # Print first 5 elements    of the first row of C

        # Measure the execution time of NumPy's built-in function
        start_time_numpy = time.time()
        C_numpy = np.dot(A, B)  # Standard matrix multiplication A @ B
        end_time_numpy = time.time()

        execution_times_numpy.append(end_time_numpy - start_time_numpy)
        print(f"NumPy Execution Time: {end_time_numpy - start_time_numpy:.6f} seconds")
        print(f"Result Shape: {C_numpy.shape}")
        print(f"Result Sample: {C_numpy[0, :5]}")  # Print first 5 elements of the first row of C


        # Store the matrix size
        matrix_sizes.append(X)

        # Check if the results are equal
        if np.allclose(C_MMM, C_numpy):
            print("Results are equal.")
            # Manually inspect inputs and outputs
            # print("A (input) sample:\n", A[:5, :5])
            # print("B (input) sample:\n", B[:5, :5])
            # print("C_MMM (output) sample:\n", C_MMM[:5, :5])
            # print("C_numpy (output) sample:\n", C_numpy[:5, :5])
        else:
            print("Results are not equal.")
            print("A (input) sample:\n", A[:5, :5])
            print("B (input) sample:\n", B[:5, :5])
            print("C_MMM (output) sample:\n", C_MMM[:5, :5])
            print("C_numpy (output) sample:\n", C_numpy[:5, :5])
            input()

    # Plot the execution times
    plt.figure(figsize=(10, 5))
    plt.plot(matrix_sizes, execution_times_MMM, label="MMM", marker="o")
    plt.plot(matrix_sizes, execution_times_numpy, label="NumPy", marker="o")
    plt.xlabel("Matrix Size (elements per side)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Matrix Multiplication Execution Time")
    plt.legend()
    plt.grid()
    plt.savefig("matrix_multiplication_execution_time.png") 