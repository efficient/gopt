#include "common.h"

__global__ void
vectorAdd(int *A, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		A[i] *= A[i];
	}
}

void gpu_run(int *h_A)
{
	int *d_A = NULL;
	int err = cudaSuccess;
	long long startCycles = 0, endCycles = 0, totCycles = 0;
	int i = 0;

	err = cudaMalloc((void **) &d_A, NUM_PKTS * sizeof(int));
	CPE(err != cudaSuccess, "Failed to cudaMalloc\n", -1);

	// Measure host-to-device memcpy latency
	for(i = 0; i < ITERS; i ++) {
		startCycles = get_cycles();
		err = cudaMemcpy(d_A, h_A, NUM_PKTS * sizeof(int), cudaMemcpyHostToDevice);
		endCycles = get_cycles();

		// 1st transfer takes a very long time
		if(i != 0) {
			totCycles += (endCycles - startCycles);
	
			printf("%d: cudaMemcpy (h2d) time: %f\n", i, (endCycles - startCycles) / 2.7);
	
			CPE(err != cudaSuccess, "Failed to copy to device memory\n", -1);
		}
	}

	long long avgCycles = totCycles / ITERS;
	double avgNanoSec = avgCycles / 2.7;
	double avgGBps = (NUM_PKTS * sizeof(int)) / avgNanoSec;
	printf("memcpy host to device stats:\n");
	printf("\tcycles = %lld, nanoseconds = %.2f ns, bandwidth = %.2f GB/s\n\n",
		avgCycles, avgNanoSec, avgGBps);

	// Measure kernel launch execution time 
	int threadsPerBlock = 256;
	int blocksPerGrid = (NUM_PKTS + threadsPerBlock - 1) / threadsPerBlock;

	totCycles = 0;
	for(i = 0; i < ITERS; i ++) {
		startCycles = get_cycles();
		vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, NUM_PKTS);
		cudaDeviceSynchronize();
		endCycles = get_cycles();
		totCycles += (endCycles - startCycles);

		if(i > ITERS / 2 && i < (ITERS / 2 + 10)) {
			printf("%d: kernel launch time: %f\n", i, (endCycles - startCycles) / 2.7);
		}
		
		err = cudaGetLastError();
		CPE(err != cudaSuccess, "Failed to launch vectorAdd kernel\n", -1);
	}

	printf("kernel launch stats:\n");
	printf("\tcycles = %lld, nanoseconds = %f ns\n\n", totCycles / ITERS,
		totCycles / (ITERS * 2.7));

	// Measure device-to-host memcpy latency
	totCycles = 0;
	for(i = 0; i < ITERS; i ++) {
		startCycles = get_cycles();
		err = cudaMemcpy(h_A, d_A, NUM_PKTS * sizeof(int), cudaMemcpyDeviceToHost);
		endCycles = get_cycles();
		totCycles += (endCycles - startCycles);

		if(i > ITERS / 2 && i < (ITERS / 2 + 10)) {
			printf("%d: cudaMemcpy (d2h) time: %f\n", i, (endCycles - startCycles) / 2.7);
		}

		CPE(err != cudaSuccess, "Failed to copy C from device to host\n", -1);
	}

	printf("memcpy device to host stats:\n");
	printf("\tcycles = %lld, nanoseconds = %f ns\n\n", totCycles / ITERS,
		totCycles / (ITERS * 2.7));
	err = cudaFree(d_A);
	CPE(err != cudaSuccess, "Failed to cudaFree\n", -1);
}

int main(void)
{
	int err = cudaSuccess;

	printDeviceProperties();

	// Allocate host vectors
	int *h_A = (int *) malloc(NUM_PKTS * sizeof(int));

	// Verify that allocations succeeded
	if (h_A == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < NUM_PKTS; ++i)	{
		h_A[i] = i;
	}

	gpu_run(h_A);
	
	// Free host memory
	free(h_A);

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

