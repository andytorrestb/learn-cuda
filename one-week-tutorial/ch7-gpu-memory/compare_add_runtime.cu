#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>


// Function to perform vector addition on the CPU
__host__ void vector_add_cpu(float *out, float *a, float *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

// Kernel function to perform vector addition on GPU
__global__ void vector_add_gpu(float *out, float *a, float *b, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        out[tid] = a[tid] + b[tid];
    }

}

int main()
{
    FILE* fp = fopen("vector_add_runtime.csv", "w");
    if (fp == NULL)
    {
        printf("Failed to open file for writing.\n");
        return 1;
    }
    else
    {
        // Write header
        fprintf(fp, "Iteration,Vector Size,GPU Time (ms),CPU Time (ms)\n");
    }

    const int n = 30; // Exponent for vector size, will be 2^n

    for (int i = 1; i <= n; i++)
    {
        printf("Iteration %d:\n", i);
        unsigned long int N = pow(2, i) - 1; // Size of the vector, doubling each iteration

        // Print the size of the vector
        printf("Vector size: %lu\n", N);

        // // Print the size of float in bytes
        // printf("Size of float: %zu bytes\n", sizeof(float));

        // Host pointers
        float *a, *b, *out;
        
        // Device pointers
        float *d_a, *d_b, *d_out;

        // Allocate host memory
        a = (float*)malloc(sizeof(float) * N);
        b = (float*)malloc(sizeof(float) * N);
        out = (float*)malloc(sizeof(float) * N);

        // Allocate device memory
        cudaMalloc((void**)&d_a, sizeof(float) * N);
        cudaMalloc((void**)&d_b, sizeof(float) * N);
        cudaMalloc((void**)&d_out, sizeof(float) * N);

        // Initialize arrays
        for (int j = 0; j < N; j++)
        {
            a[j] = (float)j;
            b[j] = (float)j;
            out[j] = 0.0f;
        }

        // Copy data from host to device
        cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


        // Number of threads per block
        // Using 64 threads per block, which follows the architecture of my GPU: NVIDIA GeForce GTX 1650
        // This can be adjusted based on the specific GPU and problem size
        int blockSize = 64;

         // Calculate number of blocks needed. Formula guarantees extra blocks if N is not a multiple of blockSize.
        int numBlocks = (N + blockSize - 1) / blockSize;
        

        printf("Number of blocks: %d\n", numBlocks);
        // Measure GPU execution time
        auto start = std::chrono::high_resolution_clock::now();
        // Launch kernel with numBlocks and blockSize threads
        vector_add_gpu<<<numBlocks, blockSize>>>(d_out, d_a, d_b, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        printf("GPU Time: %f ms\n", duration.count());
        double gpu_time = (double)duration.count();

        // Copy result from device to host
        cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

        // Print results
        // for (int i = 0; i < N; i++)
        // {
        //     printf("%f + %f = %f\n", a[i], b[i], out[i]);
        // }

        // Measure CPU execution time
        start = std::chrono::high_resolution_clock::now();
        vector_add_cpu(out, a, b, N);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        printf("CPU Time: %f ms\n", duration.count());
        double cpu_time = (double)duration.count();

        // Write the results to the CSV file
        fprintf(fp, "%d,%lu,%f,%f\n", i, N, gpu_time, cpu_time);

        // Free device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        // Free host memory
        free(a);
        free(b);
        free(out);

    }

    fclose(fp);
    printf("Results written to vector_add_runtime.csv\n");

    return 0;
}
