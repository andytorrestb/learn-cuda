/*
This script demonstrates the use of device functions in CUDA.
It includes device functions that print messages, a kernel that calls these functions,
and a host function that launches the kernel.

It also shows how to synchronize the device and host, mainly using `cudaDeviceSynchronize()`.

Try uncommenting the `cudaDeviceSynchronize()` calls in various places to see how it affects the output.

*/

#include <stdio.h>

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
    Device2();
}

__host__ void sub_Function_in_Host()
{
    kernel<<<1, 1>>>();
    // cudaDeviceSynchronize();
}

int main()
{
    // cudaDeviceSynchronize();
    sub_Function_in_Host();
    // cudaDeviceSynchronize();
    sub_Function_in_Host();
    // cudaDeviceSynchronize();
    return 0;
}