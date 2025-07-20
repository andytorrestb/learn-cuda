#include <stdio.h>

#define N 100

__global__ void vector_add(float *out, float*a, float *b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		out[i] = a[i] + b[i];
	}
}

int main()
{
	// Host and device pointers.
	// Here device represents the GPU memory.
	// Host memory is the CPU memory.
	float *a, *b, *out;
	float *d_a, *d_b, *d_out;

	// Allocate host memory
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	// Allocate device memory
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_out, sizeof(float) * N);

	// Initialize array
	for (int i = 0; i < N; i++)
	{
		a[i] = 1.0f; b[i] = 2.0f; 
		//out[i] = 0.0f;
	}

	// Copy data from host to device
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);

	// Launch kernel
	vector_add<<<1,N>>>(d_out, d_a, d_b, N);

	// Copy result from device to host
	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	// Print results
	for (int i = 0; i < N; i++)
	{
		printf("%f+%f = %f\n", a[i], b[i], out[i]);
	}


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
