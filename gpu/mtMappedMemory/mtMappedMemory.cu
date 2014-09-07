#include "common.h"

#define NUM_THREADS 1
pthread_t cpu_thread[NUM_THREADS];		// CPU threads that talk to the GPU
int tid[NUM_THREADS];

int volatile *h_A, *h_B;
int volatile *d_A, *d_B;
int volatile *h_flag, *d_flag;
int volatile *d_log = NULL;				// Host does not access the log

int NUM_PKTS = 32;						// Packets per CPU thread

__global__ void
vectorAdd(volatile int *A, volatile int *B, volatile int *flag, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int iter = 0;
	int cpu_tid = i / 32;

	// Index in flag array to poll on - staggered to avoid false sharing
	int flag_i = cpu_tid * CACHELINE_CAP;
	
	if(i < N) {
		// Wait for CPU flag a finite number of times
		for(iter = 0; iter < ITERS; iter ++) {
			while(flag[flag_i] == iter - 1) {
				// Wait for host flag to be raised
			}

			B[i] = A[i] * 2;
		}
	}
}


void *gpu_run(void *ptr)
{
	int tid = *((int *) ptr);
	int flag_i = tid * CACHELINE_CAP;

	// Full execution measurements
	long long start_cycles = 0, end_cycles = 0, tot_cycles = 0;

	int iter = 0, j = 0;

	for(iter = 0; iter < ITERS; iter ++) {
		printf("Thread %d, iteration %d\n", tid, iter);
		memset((char *) h_B, 0, NUM_PKTS * sizeof(int));

		start_cycles = get_cycles();
		
		for(j = 0; j < NUM_PKTS; j ++) {
			h_A[j] = (iter & 0xff) + j + 1;		// Always > 0
			assert(h_A[j] != 0);
		}

		// XXX: Do we need a memory barrier here? h_A and h_flag are volatile..

		// Raise a flag for the GPU
		h_flag[flag_i] = iter;

		// Wait till the GPU makes all of h_B non-zero
		waitForNonZero(h_B, NUM_PKTS, tid);

		for(j = 0; j < NUM_PKTS; j ++) {
			if(h_B[j] != h_A[j] * 2) {
				fprintf(stderr, "Kernel output mismatch error\n");
				exit(-1);
			}
		}

		end_cycles = get_cycles();

		tot_cycles += (end_cycles - start_cycles);

		printf("\tThread %d, iter %d: %d ns\n", tid, iter, (int) ((end_cycles - start_cycles) / 2.7));
	}

	printf("\nFull execution stats: %d ns\n", (int) (tot_cycles / (2.7 * ITERS)));

	return 0;
}

int main(int argc, char **argv)
{
	assert(argc == 1);

	int err = cudaSuccess;
	printDeviceProperties();

	cudaSetDeviceFlags(cudaDeviceMapHost);

	// Allocate host vectors as mapped memory
	err = cudaHostAlloc(&h_A, NUM_PKTS * NUM_THREADS * sizeof(int), cudaHostAllocMapped);
	err = cudaHostAlloc(&h_B, NUM_PKTS * NUM_THREADS * sizeof(int), cudaHostAllocMapped);

	// Allocate the flag array (host memory version): one cacheline per flag (no false sharing)
	err = cudaHostAlloc(&h_flag, CACHELINE_CAP * NUM_THREADS * sizeof(int), cudaHostAllocMapped);

	CPE(err != cudaSuccess, "Could not allocate managed memory for packets\n", -1);

	// Zero out the mapped memory packet containers
	assert(h_A != NULL);
	assert(h_B != NULL);
	assert(h_flag != NULL);

	for(int j = 0; j < NUM_PKTS; j++) {
		h_A[j] = 0;
		h_B[j] = 0;
	}

	// The kernel expects that for iteration #i, i >= 0, the flag will be i
	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		int flag_i = thread_i * CACHELINE_CAP;
		h_flag[flag_i] = -1;
	}
		
	// Get device pointers for mapped memory
	err = cudaHostGetDevicePointer((void **) &d_A, (void *) h_A, 0);
	err = cudaHostGetDevicePointer((void **) &d_B, (void *) h_B, 0);

	err = cudaHostGetDevicePointer((void **) &d_flag, (void *) h_flag, 0);

	CPE(err != cudaSuccess, "Could not get device pointer for mapped memory\n", -1);

	// Launch the CPU threads
	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		tid[thread_i] = thread_i;
		pthread_create(&cpu_thread[thread_i], NULL, gpu_run, &tid[thread_i]);
	}

	// Launch the kernel once
	printf("Launching CUDA kernel\n");
	int threadsPerBlock = NUM_PKTS;
	int totalThreads = (NUM_PKTS * NUM_THREADS);
	assert(threadsPerBlock <= 256);
	int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

	cudaStream_t my_stream;
	err = cudaStreamCreate(&my_stream);
	CPE(err != cudaSuccess, "Failed to create cudaStream\n", -1);

	vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_A, d_B, d_flag, NUM_PKTS * NUM_THREADS);
	cudaStreamQuery(my_stream);

	printf("Waiting for CPU threads to finish\n");
	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		pthread_join(cpu_thread[thread_i], NULL);
	}

	// Free allocated mapped memory
	printf("Freeing mapped memory\n");
	cudaFreeHost((void *) h_A);
	cudaFreeHost((void *) h_B);
	cudaFreeHost((void *) h_flag);

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

