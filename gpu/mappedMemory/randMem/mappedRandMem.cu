#include <cuda_runtime.h>
#include "common.h"

/** < Use global variables because it's gpu_run is a separate thread */
int volatile *h_A, *h_B;
int volatile *d_A, *d_B;
int volatile *h_flag, *d_flag;
int *h_log, *d_log;					// Not in mapped memory 
int num_pkts = -1;					// Passed as a command line argument

pthread_t cpu_thread;				// CPU thread that talks to the GPU

__global__ void
vectorAdd(volatile int *A, volatile int *B, volatile int *flag, int *log, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int iter = 0;
	
	if(i < N) {
		// Wait for CPU flag a finite number of times
		for(iter = 0; iter < ITERS; iter ++) {
			while(flag[0] == iter - 1) {
				// Wait for host flag to be raised
			}

			int address = A[i];

			B[i] = log[address];
		}
	}
}


void *gpu_run(void *ptr)
{
	struct timespec start, end;
	double tot = 0, iter_us;

	int iter = 0, j = 0, sum = 0;

	for(iter = 0; iter < ITERS; iter ++) {
		memset((char *) h_B, 0, num_pkts * sizeof(int));

		/** < Write input addresses into A */
		for(j = 0; j < num_pkts; j ++) {
			h_A[j] = rand() & LOG_CAP_;
		}

		/** < Start a timer */
		clock_gettime(CLOCK_REALTIME, &start);

		/** < Raise a flag for the GPU */
		h_flag[0] = iter;

		/** < Wait till the GPU makes all of h_B non-zero */
		waitForNonZero(h_B, num_pkts);

		clock_gettime(CLOCK_REALTIME, &end);

		for(j = 0; j < num_pkts; j ++) {
			if(h_B[j] != h_log[h_A[j]]) {
				fprintf(stderr, "Kernel output mismatch error\n");
				exit(-1);
			}
			sum += h_B[j];
		}

		iter_us = (end.tv_sec - start.tv_sec) * 1000000 + 
			(double) (end.tv_nsec - start.tv_nsec) / 1000;

		printf("\tIter %d: %.2f us. Per packet: %.2f ns, sum = %d\n", 
			iter, iter_us, iter_us * 1000 / num_pkts, sum);

		tot += iter_us;
	}

	printf("Average %.2f us\n", tot / ITERS);

	return 0;
}

int main(int argc, char **argv)
{
	assert(argc == 2);
	num_pkts = atoi(argv[1]);

	int err = cudaSuccess;
	printDeviceProperties();

	cudaSetDeviceFlags(cudaDeviceMapHost);

	/** < Allocate host vectors as mapped memory */
	err = cudaHostAlloc(&h_A, num_pkts * sizeof(int), cudaHostAllocMapped);
	err = cudaHostAlloc(&h_B, num_pkts * sizeof(int), cudaHostAllocMapped);
	err = cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped);
	CPE(err != cudaSuccess, "Could not allocate managed memory\n", -1);

	assert(h_A != NULL);
	assert(h_B != NULL);
	assert(h_flag != NULL);

	// The kernel expects that for iteration #i, i >= 0, the flag will be i
	h_flag[0] = -1;

	// Zero out the mapped memory vectors
	for(int i = 0; i < num_pkts; i++) {
		h_A[i] = 0;
		h_B[i] = 0;
	}

	/** < Allocate a non-zero log in normal host memory and copy it over*/
	h_log = (int *) malloc(LOG_CAP * sizeof(int));
	// Add some data to the log. Log data should be non-zero
	for(int i = 0; i < LOG_CAP; i ++) {
		h_log[i] = i + 1;
	}

	err = cudaMalloc((void **) &d_log, LOG_CAP * sizeof(int));
	CPE(err != cudaSuccess, "Could not alloc log on device\n", -1);

	err = cudaMemcpy(d_log, h_log, LOG_CAP * sizeof(int), 
		cudaMemcpyHostToDevice);
	CPE(err != cudaSuccess, "Could not copy log to device\n", -1);

	// Get device pointer for mapped memory
	err = cudaHostGetDevicePointer((void **) &d_A, (void *) h_A, 0);
	err = cudaHostGetDevicePointer((void **) &d_B, (void *) h_B, 0);
	err = cudaHostGetDevicePointer((void **) &d_flag, (void *) h_flag, 0);

	CPE(err != cudaSuccess, "Could not get device pointer for mapped memory\n", -1);

	// Launch the CPU code
	pthread_create(&cpu_thread, NULL, gpu_run, NULL);

	// Launch the kernel once
	printf("Launching CUDA kernel\n");
	int threadsPerBlock = 256;
	int blocksPerGrid = (num_pkts + threadsPerBlock - 1) / threadsPerBlock;

	cudaStream_t my_stream;
	err = cudaStreamCreate(&my_stream);
	CPE(err != cudaSuccess, "Failed to create cudaStream\n", -1);

	vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_A, d_B, d_flag, d_log, num_pkts);
	cudaStreamQuery(my_stream);

	printf("Waiting for CPU thread to finish\n");
	pthread_join(cpu_thread, NULL);

	// Free allocated mapped memory
	cudaFreeHost((void *) h_A);
	cudaFreeHost((void *) h_B);
	cudaFreeHost((void *) h_flag);

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

