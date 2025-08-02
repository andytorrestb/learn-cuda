#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Size of the vector
#define N 10000000

// Kernel function to perform vector addition
__global__ void vector_add_gpu(float *out, float *a, float *b, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        out[tid] = a[tid] + b[tid];
    }

}

__host__ void vector_add_cpu(float *out, float *a, float *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

int main()
{
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
    for (int i = 0; i < N; i++)
    {
        a[i] = (float)i;
        b[i] = (float)i;
        out[i] = 0.0f;
    }

    // Copy data from host to device
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


    int blockSize = 256; // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; //
    // Launch kernel with 1 block and N threads
    // Measure GPU execution time
    auto start = std::chrono::high_resolution_clock::now();
    vector_add_gpu<<<numBlocks, blockSize>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    printf("GPU Time: %f ms\n", duration.count());

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

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Free host memory
    free(a);
    free(b);
    free(out);

    return 0;
}
