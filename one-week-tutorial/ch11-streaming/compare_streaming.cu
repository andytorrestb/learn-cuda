#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

// Convinienct function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in reslease builds.
inline cudaError_t checkCuda(cudaError_t result)
{
    #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    #endif
    return result;
}

__global__ void kernel(float *a, int offset)
{
    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)i;
    float s = sinf(x);
    float c = cosf(x);
    a[i] = a[i] + sqrtf(s*s + c*c);
}

float maxError(float *a, int n)
{
    float mexE = 0.0;
    for (int i = 0; i < n; i++)
    {
        float error = fabs(a[i] - 1.0f);
        if (error > mexE)
        {
            mexE = error;
        }
    }
    return mexE;
}

int main(int argc, char **argv)
{
    FILE* fp = fopen("compare_streaming_runtime.csv", "w");
    if (fp == NULL)
    {
        printf("Failed to open file for writing.\n");
        return 1;
    }
    else
    {
        // Write header
        fprintf(fp, "Iteration,Vector Size,Streams,Baseline Time (ms), Async1 Time (ms), Async2 Time (ms)\n");
    }

    const int N = 28;
    for (int j = 1; j <= N; j++)
    {
        for (int k = 1; k <= 5; k++)
        {

            int blockSize = 64, nStreams = 1 << k; // 1, 2, 4, 8
            int n  = 1 << j; // 2^j
            int streamSize = n / nStreams;
            int streamBytes = streamSize * sizeof(float);
            int bytes = n * sizeof(float);
            
            printf("\nUsing %d streams\n", nStreams);
            printf("Running analysis for vector size: %d\n", n);

            // Allocate pinned host memory and device memory
            float *h_a, *d_a;
            checkCuda(cudaMallocHost((void**)&h_a, bytes)); // Host pinned
            checkCuda(cudaMalloc((void**)&d_a, bytes)); // Device memory
        
            float ms; // Elapsed time in milliseconds
        
            // Create event and streams
            cudaStream_t streams[nStreams];
        
            cudaEvent_t startEvent, stopEvent;
            checkCuda(cudaEventCreate(&startEvent));
            checkCuda(cudaEventCreate(&stopEvent));
        
            for (int i = 0; i < nStreams; i++)
            {
                checkCuda(cudaStreamCreate(&streams[i]));
            }
        
            // Baseline case - sequential transfer and execution
            memset(h_a, 0, bytes);
            checkCuda(cudaEventRecord(startEvent, 0));
            checkCuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
            kernel<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_a, 0);
            checkCuda(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
            checkCuda(cudaEventRecord(stopEvent, 0));
            checkCuda(cudaEventSynchronize(stopEvent));
            checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
            printf("Elapsed time for sequential transfer and execution: %.2f ms\n", ms);
            printf("Max error: %.6f\n", maxError(h_a, n));
            float baseline_time = ms;
        
            // Asychronous version 1: loop over {copy, kernel, copy}
            memset(h_a, 0, bytes);
            checkCuda(cudaEventRecord(startEvent, 0));
            for (int i = 0; i < nStreams; i++)
            {
                int offset = i * streamSize;
                
                checkCuda
                (
                    cudaMemcpyAsync
                    (
                        &d_a[offset],
                        &h_a[offset],
                        streamBytes,
                        cudaMemcpyHostToDevice,
                        streams[i]
                    )
                );
        
                int nBlocks = (streamSize + blockSize - 1) / blockSize;
                kernel<<<nBlocks, blockSize, 0, streams[i]>>>(d_a, offset);
        
                checkCuda
                (
                    cudaMemcpyAsync
                    (
                        &h_a[offset],
                        &d_a[offset],
                        streamBytes,
                        cudaMemcpyDeviceToHost,
                        streams[i]
                    )
                );
            }
        
            checkCuda(cudaEventRecord(stopEvent, 0));
            checkCuda(cudaEventSynchronize(stopEvent));
            checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
            printf("Elapsed time for async transfer and execution (version 1): %.2f ms\n", ms);
            printf("Max error: %.6f\n", maxError(h_a, n));
            float async1_time = ms;
        

            // Asychronous version 2: loop over {all copies, all kernels, all copies}
            memset(h_a, 0, bytes);
            checkCuda(cudaEventRecord(startEvent, 0));
        
            // First, copy all data from host to device
            for (int i = 0; i < nStreams; i++)
            {
                int offset = i * streamSize;
                checkCuda
                (
                    cudaMemcpyAsync
                    (
                        &d_a[offset],
                        &h_a[offset],
                        streamBytes,
                        cudaMemcpyHostToDevice,
                        streams[i]
                    )
                );
            }
        
            // Second, launch all kernels
            for (int i = 0; i < nStreams; i++)
            {
                int offset = i * streamSize;
                int nBlocks = (streamSize + blockSize - 1) / blockSize;
                kernel<<<nBlocks, blockSize, 0, streams[i]>>>(d_a, offset);
            }
        
            // Finally, copy all results back to host
            for (int i = 0; i < nStreams; i++)
            {
                int offset = i * streamSize;
                checkCuda
                (
                    cudaMemcpyAsync
                    (
                        &h_a[offset],
                        &d_a[offset],
                        streamBytes,
                        cudaMemcpyDeviceToHost,
                        streams[i]
                    )
                );
            }
        
            checkCuda(cudaEventRecord(stopEvent, 0));
            checkCuda(cudaEventSynchronize(stopEvent));
            checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
            printf("Elapsed time for async transfer and execution (version 2): %.2f\n", ms);
            printf("Max error: %.6f\n", maxError(h_a, n));
            float async2_time = ms;
        
            // Cleanup
            checkCuda(cudaEventDestroy(startEvent));
            checkCuda(cudaEventDestroy(stopEvent));
            for (int i = 0; i < nStreams; i++)
            {
                checkCuda(cudaStreamDestroy(streams[i]));
            }
        
            checkCuda(cudaFreeHost(h_a));
            checkCuda(cudaFree(d_a));

            // Write the results to the CSV file
            fprintf(fp, "%d,%d,%d,%f,%f,%f\n", j, n, nStreams, baseline_time, async1_time, async2_time);
        }
    }

    fclose(fp);
    printf("Results written to compare_streaming_runtime.csv\n");

    return 0;
}