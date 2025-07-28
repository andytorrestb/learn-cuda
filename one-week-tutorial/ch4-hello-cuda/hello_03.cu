/*
This script demonstrates the use of device functions in CUDA.
It includes device functions that print messages, a kernel that calls these functions,
and a host function that launches the kernel.

It also shows how to synchronize the device and host, mainly using `cudaDeviceSynchronize()`.

Try uncommenting the `cudaDeviceSynchronize()` calls in various places to see how it affects the output.

*/

#include <stdio.h>
#include <chrono>

__device__ void Device1()
{
    printf("Hello from Device1\n");
    // cudaDeviceSynchronize(); // will cause an error: host function cannot be called from a device function
}

__device__ void Device2()
{
    printf("Hello from Device2\n");
    // cudaDeviceSynchronize(); // same error as above
}

__global__ void kernel()
{   
    Device1();
    // cudaDeviceSynchronize(); // will cause an error: host function cannot be called from a global (kernel) function
    // Device2();
}

__host__ void sub_Function_in_Host(int M, int T)
{
    kernel<<<M, T>>>();
}

int main()
{

    // Initialize a 2D array to store run time results
    int rows = 100;
    int cols = 100;
    double** run_time = (double**)malloc(rows * sizeof(double*));
    if (run_time == NULL) {
        printf("Failed to allocate memory for run_time array.\n");
        return 1;
    }

    for (int i = 0; i < rows; i++) {
        // Allocate memory for each row
        run_time[i] = (double*)malloc(cols * sizeof(double));
        if (run_time[i] == NULL) {
            printf("Failed to allocate memory for run_time[%d].\n", i);
            // Free previously allocated memory
            for (int j = 0; j < i; j++) free(run_time[j]);
            free(run_time);
            return 1;
        }
    }

    // Measure execution time for different configurations
    // A nested loop is used to vary the number of 
    // blocks (M) and threads (T)
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            auto start = std::chrono::high_resolution_clock::now();
        
            sub_Function_in_Host(i, j);
            cudaDeviceSynchronize();
    
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            printf("Execution time: %f ms\n", duration.count());
    
            run_time[i][j] = (double)duration.count();
        }
    }

    // Write the run time data to a CSV file
    FILE* fp = fopen("runtime_data.csv", "w");
    if (fp == NULL) {
        printf("Failed to open file for writing.\n");
    } else {
        // Write header
        fprintf(fp, "i,j,run_time_ms\n");
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                fprintf(fp, "%d,%d,%f\n", i, j, run_time[i][j]);
            }
        }
        fclose(fp);
        printf("Runtime data written to runtime_data.csv\n");
    }

    free(run_time);
    return 0;
}