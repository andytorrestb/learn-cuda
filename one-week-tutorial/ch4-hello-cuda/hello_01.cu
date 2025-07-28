/*

This script demonstrates the use of device functions in CUDA to create a standard "Hello World" example.

*/

#include <stdio.h>

__device__ void Device1()
{
    printf("Hello from Device1\n");
}

__device__ void Device2()
{
    printf("Hello from Device2\n");
}

__global__ void kernel()
{
    Device1();
    Device2();
}

__host__ void sub_Function_in_Host()
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main()
{
    sub_Function_in_Host();
    return 0;
}