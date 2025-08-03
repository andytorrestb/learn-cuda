#include <cassert>
#include <iostream>
#include <chrono>

using std::cout;
using std::end;

__global__ void vectorAdd(int *a, int *b, int *c, int N)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

void verify_result(int *a, int *b, int *c, int N)
{
    for (int i = 0; i < N; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

void initiaze_array(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        a[i] = 1;
    }
}

int main()
{
    FILE* fp = fopen("pinned_memory_runtime.csv", "w");
    if (fp == NULL)
    {
        printf("Failed to open file for writing.\n");
        return 1;
    }
    else
    {
        // Write header
        fprintf(fp, "Iteration,Vector Size,Pinned Memory Time (ms),Paged Memory Time (ms)\n");
    }

    const int n = 28; // Exponent for vector size, will be 2^n

    for (int i = 1; i <= n; i++)
    {
        cout << "Running analysis for n = " << i << "\n";
        
        // Your code here
        int N = 1 << i; // 2^i
        size_t bytes = N * sizeof(int);
    
        // Vectors for holding host-side (CPU-side) data
        int *h_a;
    
        // Allocate pinned memory on the host
        cudaMallocHost(&h_a, bytes);
    
    
        // Initialize the host array
        initiaze_array(h_a, N);
    
        // Allocate memory on the device (GPU)
        int *d_a;
        cudaMalloc(&d_a, bytes);
    
    
        // Copy data from host to device (CPU -> GPU)
        auto start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        printf("Data transfer time (pinned): %f ms\n", duration.count());
        float pinned_time = duration.count();

        // Free pinned host memory
        cudaFreeHost(h_a);
    
    
        // Free device memory
        cudaFree(d_a);
    
    
        // cout << "SUCCESSFULLY COMPLETED PINNED DATA TRANSFER!\n";
    
        // Allocate pinned memory on the host
        h_a = (int *)malloc(bytes);
    
    
        // Initialize the host array
        initiaze_array(h_a, N);
    
        // Allocate memory on the device (GPU)
        cudaMalloc(&d_a, bytes);
    
    
        // Copy data from host to device (CPU -> GPU)
        start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        printf("Data transfer time (paged): %f ms\n", duration.count());
        float paged_time = duration.count();
    
        // Free pinned host memory
        cudaFreeHost(h_a);
    
    
        // Free device memory
        cudaFree(d_a);

        // Write the results to the CSV file
        fprintf(fp, "%d,%d,%f,%f\n", i, N, pinned_time, paged_time);
    }

    fclose(fp);
    printf("Pinned memory analysis completed and results written to pinned_memory_runtime.csv\n");
    return 0;
}