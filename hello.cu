//===================================
// CUDA Hello World
//==================================
// Simple program demonstrates launching a GPU kernel
// that prints a message from each thread.

// It is  meant to serve as a sanity check and test for 
// proper CUDA installation and environment set up.


#include <stdio.h>


// Device (GPU) Kernel Function
__global__ void cuda_hello()
{
    	// Each thread on the GPU will execute this line in parallel.
	printf("Hello World from your GPU!\n");
	return;
}


// Host (CPU) Main function
int main()
{
	// Define grid and block dimensions:
	// M blocks, each with T threads
	int M = 1;
	int T = 10;

	// Launches kernel on the GPU
	// The <<<M, T>>> syntax is CUDA's kernel launch configuration:
	// - M specifies the number of blocks in the grid
	// - T is the number of threads per block
	// This results in M*T total threads to run cuda_hello() in parallel.
	cuda_hello<<<M,T>>>();

	// Waits for GPU to finish before the program exits.
	// This ensures the printf outputs are flushed to the console.
	cudaDeviceSynchronize();

	return 0;

}
