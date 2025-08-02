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
    printf("Hello from Device1\n");}

__global__ void kernel()
{   
    Device1();
}

__host__ void sub_Function_in_Host(int M, int T)
{
    kernel<<<M, T>>>();
}

int main()
{

    auto global_start = std::chrono::high_resolution_clock::now();

    // Initialize a 2D array to store run time results
    int block_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int grid_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    int num_grid_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    double** run_time = (double**)malloc(num_grid_sizes * sizeof(double*));
    if (run_time == NULL) {
        printf("Failed to allocate memory for run_time array.\n");
        return 1;
    }

    for (int i = 0; i < num_grid_sizes; i++) {
        // Allocate memory for each row
        run_time[i] = (double*)malloc(num_block_sizes * sizeof(double));
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
    for (int i = 0; i < num_grid_sizes; i++)
    {
        for (int j = 0; j < num_block_sizes; j++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            sub_Function_in_Host(grid_sizes[i], block_sizes[j]);
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
        fprintf(fp, "grid_size,block_size,run_time_ms\n");
        for (int i = 0; i < num_grid_sizes; i++) {
            for (int j = 0; j < num_block_sizes; j++) {
                
                // Write each row of data
                fprintf
                (
                    fp, "%d,%d,%f\n", 
                    grid_sizes[i], block_sizes[j], run_time[i][j]
                );
            }
        }
        fclose(fp);
        printf("Runtime data written to runtime_data.csv\n");
    }

    free(run_time);

    auto global_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> global_duration = global_end - global_start;
    printf("Total execution time: %f ms\n", global_duration.count());

    return 0;
}