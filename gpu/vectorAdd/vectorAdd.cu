#include "common.h"

__global__ void
vectorAdd(const int *A, const int *B, int *C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		C[i] = 0;

		int j;
		for(j = 0; j < COMPUTE; j ++) {
			C[i] += A[i] + B[i];
			C[i] *= C[i];
		}
	}
}

void cpu_run(int *A, int *B, int *C, int N)
{
	int i;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for(i = 0; i < N; i ++) {
		int j;
		for(j = 0; j < COMPUTE; j ++) {
			C[i] += A[i] + B[i];
			C[i] *= C[i];
		}
	}

	clock_gettime(CLOCK_REALTIME, &end);
	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);
	printf("CPU time = %f\n", time);
}

void gpu_run(int *h_A, int *h_B, int *h_C, int N)
{
	struct timespec start, end;
	int *d_A = NULL, *d_B = NULL, *d_C = NULL;
	int err = cudaSuccess;

	err = cudaMalloc((void **) &d_A, N * sizeof(int));
	err = cudaMalloc((void **) &d_B, N * sizeof(int));
	err = cudaMalloc((void **) &d_C, N * sizeof(int));
	CPE(err != cudaSuccess, "Failed to cudaMalloc\n", -1);

	err = cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);
	CPE(err != cudaSuccess, "Failed to copy to device memory\n", -1);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	// Start the clock
	clock_gettime(CLOCK_REALTIME, &start);

	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_REALTIME, &end);
	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);
	printf("GPU time = %f\n", time);

	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch vectorAdd kernel\n", -1);

	// Copy back the result
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
	CPE(err != cudaSuccess, "Failed to copy C from device to host\n", -1);

	// Free device global memory
	err = cudaFree(d_A);
	err = cudaFree(d_B);
	err = cudaFree(d_C);
}

int main(void)
{
	int err = cudaSuccess, N = 500000;

	printDeviceProperties();

	printf("[Vector addition of %d elements]\n", N);

	// Allocate host vectors
	int *h_A = (int *) malloc(N * sizeof(int));
	int *h_B = (int *) malloc(N * sizeof(int));
	int *h_C = (int *) malloc(N * sizeof(int));
	int *h_C_CPU = (int *) malloc(N * sizeof(int));

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < N; ++i)	{
		h_A[i] = rand() / (int) RAND_MAX;
		h_B[i] = rand() / (int) RAND_MAX;
	}

	cpu_run(h_A, h_B, h_C_CPU, N);
	gpu_run(h_A, h_B, h_C, N);

	// Verify that the result vector is correct
	for (int i = 0; i < N; ++i) {
		if (fabs(h_C_CPU[i] - h_C[i]) > 1) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			fprintf(stderr, "CPU %d, GPU %d\n", h_C_CPU[i], h_C[i]);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");

	CPE(err != cudaSuccess, "Failed to cudaFree\n", -1);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_CPU);

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

