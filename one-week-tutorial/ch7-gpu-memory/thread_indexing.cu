#include <stdio.h>
#include <stdlib.h>

__global__ void kernel()
{
    int temp = 0;
    temp = threadIdx.x;

    printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d, temp: %d\n",
           blockIdx.x, blockDim.x, threadIdx.x, temp);

}

__global__ void kernel2()
{
    int temp = 0;
    temp = blockIdx.x * blockDim.x + threadIdx.x;

    printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d, temp: %d\n",
           blockIdx.x, blockDim.x, threadIdx.x, temp);
}

int main()
{
    // Q: Why is this kernel function only printing threadIdx.x values from 0 to 4?
    // When the expected output should ranging from 0 to 24?

    // A: Because each block exists in its own memory space (thus resetting the threadIdx.x values for each block).
    // How can this be fixed?
    kernel<<<5, 5>>>();
    cudaDeviceSynchronize();


    printf("=================================================\n");


    // This can be fixed by calculating the global thread index
    // using the formula: blockIdx.x * blockDim.x + threadIdx.x
    kernel2<<<5, 5>>>();
    cudaDeviceSynchronize();

    return 0;
}


// The kernel functions should be re-named to reflect their purpose (and adhere to readability conentions).
// The first kernel can be named `printLocalThreadIdx` and the second one can be named `printGlobalThreadIdx`.
// However, the current names are kept for demonstration purposes.
