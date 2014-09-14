#include "common.h"

int volatile *h_A, *h_B;
int volatile *d_A, *d_B;
int volatile *h_flag, *d_flag;
int volatile *h_log, *d_log;			// Host does not access the log
int NUM_PKTS = -1;					// Passed as a command line argument

pthread_t cpu_thread;				// CPU thread that talks to the GPU

__global__ void
vectorAdd(volatile int *A, volatile int *B, volatile int *flag, volatile int *log, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int iter = 0;
	
	if(i < N) {
		// Wait for CPU flag a finite number of times
		for(iter = 0; iter < ITERS; iter ++) {
			while(flag[0] == iter - 1) {
				// Wait for host flag to be raised
			}

			B[i] = log[i * LOG_STEP] + iter;
		}
	}
}


void *gpu_run(void *ptr)
{
	// Full execution measurements
	long long start_cycles = 0, end_cycles = 0, tot_cycles = 0;

	int iter = 0, j = 0;

	for(iter = 0; iter < ITERS; iter ++) {
		printf("Iteration %d\n", iter);
		memset((char *) h_B, 0, NUM_PKTS * sizeof(int));

		start_cycles = get_cycles();
		
		for(j = 0; j < NUM_PKTS; j ++) {
			h_A[j] = (iter & 0xff) + j + 1;		// Always > 0
			assert(h_A[j] != 0);
		}

		// XXX: Do we need a memory barrier here? h_A and h_flag are volatile

		// Raise a flag for the GPU
		h_flag[0] = iter;

		// Wait till the GPU makes all of h_B non-zero
		waitForNonZero(h_B, NUM_PKTS);

		for(j = 0; j < NUM_PKTS; j ++) {
			if(h_B[j] != h_log[j * LOG_STEP] + iter) {
				fprintf(stderr, "Kernel output mismatch error\n");
				exit(-1);
			}
		}

		end_cycles = get_cycles();

		tot_cycles += (end_cycles - start_cycles);

		printf("\tIter %d: %d ns\n", iter, (int) ((end_cycles - start_cycles) / 2.7));
	}

	printf("\nFull execution stats: %d ns\n", (int) (tot_cycles / (2.7 * ITERS)));

	return 0;
}

int main(int argc, char **argv)
{
	assert(argc == 2);

	NUM_PKTS = atoi(argv[1]);
	assert(NUM_PKTS > 0 && NUM_PKTS <= 256);
	assert(NUM_PKTS * LOG_STEP < LOG_CAP);

	int err = cudaSuccess;
	printDeviceProperties();

	cudaSetDeviceFlags(cudaDeviceMapHost);

	// Allocate host vectors as mapped memory
	err = cudaHostAlloc(&h_A, NUM_PKTS * sizeof(int), cudaHostAllocMapped);
	err = cudaHostAlloc(&h_B, NUM_PKTS * sizeof(int), cudaHostAllocMapped);
	err = cudaHostAlloc(&h_log, LOG_CAP * sizeof(int), cudaHostAllocMapped);

	// Allocate the flag (host memory version)
	err = cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped);
		
	CPE(err != cudaSuccess, "Could not allocate managed memory\n", -1);

	assert(h_A != NULL);
	assert(h_B != NULL);
	assert(h_log != NULL);
	assert(h_flag != NULL);

	// The kernel expects that for iteration #i, i >= 0, the flag will be i
	h_flag[0] = -1;

	// Zero out the mapped memory vectors
	for(int i = 0; i < NUM_PKTS; i++) {
		h_A[i] = 0;
		h_B[i] = 0;
	}

	// Add some data to the log
	for(int i = 0; i < LOG_CAP; i ++) {
		h_log[i] = i + 1;
	}

	// Get device pointer for mapped memory
	err = cudaHostGetDevicePointer((void **) &d_A, (void *) h_A, 0);
	err = cudaHostGetDevicePointer((void **) &d_B, (void *) h_B, 0);
	err = cudaHostGetDevicePointer((void **) &d_flag, (void *) h_flag, 0);
	err = cudaHostGetDevicePointer((void **) &d_log, (void *) h_log, 0);

	CPE(err != cudaSuccess, "Could not get device pointer for mapped memory\n", -1);

	// Launch the CPU code
	pthread_create(&cpu_thread, NULL, gpu_run, NULL);

	// Launch the kernel once
	printf("Launching CUDA kernel\n");
	int threadsPerBlock = NUM_PKTS;
	assert(threadsPerBlock <= 256);

	int blocksPerGrid = (NUM_PKTS + threadsPerBlock - 1) / threadsPerBlock;

	cudaStream_t my_stream;
	err = cudaStreamCreate(&my_stream);
	CPE(err != cudaSuccess, "Failed to create cudaStream\n", -1);

	vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_A, d_B, d_flag, d_log, NUM_PKTS);
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

