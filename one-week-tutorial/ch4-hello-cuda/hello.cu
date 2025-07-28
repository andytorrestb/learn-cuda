#include <stdio.h>

__device__ void Device1()
{
    //
}

__device__ void Device2()
{
    //
}

__global__ void kernel()
{
    Device1();
    Device2();
}

void sub_Function_in_Host()
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main()
{
    sub_Function_in_Host();
    return 0;
}