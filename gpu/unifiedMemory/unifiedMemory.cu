#include "common.h"

#define NUM_THREADS 1

int *A[NUM_THREADS], *B[NUM_THREADS];
int tid[NUM_THREADS];
pthread_t thread[NUM_THREADS];

__global__ void
vectorAdd(int *A, int *B, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		B[i] = A[i] * A[i];
	}
}

void *gpu_run(void *ptr)
{
	int tid = *((int *) ptr);
	printf("My tid = %d\n", tid);

	cudaStream_t my_stream;
		
	int err = cudaSuccess;

	// Per stage measurements
	long long start_cycles_h2d = 0, start_cycles_kernel = 0, start_cycles_d2h = 0;
	long long end_cycles_h2d = 0, end_cycles_kernel = 0, end_cycles_d2h = 0;
	long long tot_cycles_h2d = 0, tot_cycles_kernel = 0, tot_cycles_d2h = 0;

	// Full execution measurements
	long long start_cycles = 0, end_cycles = 0, tot_cycles = 0;

	int i = 0, j = 0;

	int threadsPerBlock = 16;
	int blocksPerGrid = (NUM_PKTS + threadsPerBlock - 1) / threadsPerBlock;

	err = cudaStreamCreate(&my_stream);
	CPE(err != cudaSuccess, "Failed to create cudaStream\n", -1);

	for(i = 0; i < ITERS; i ++) {
		start_cycles = get_cycles();
		
		// Stage 1: host to device latency
		start_cycles_h2d = get_cycles();
		for(j = 0; j < NUM_PKTS; j ++) {
			A[tid][j] = (i & 0xff) + j;
		}
		end_cycles_h2d = get_cycles();

		// Stage 2: kernel execution latency
		start_cycles_kernel = get_cycles();
		vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(A[tid], B[tid], NUM_PKTS);
		end_cycles_kernel = get_cycles();

		// Complete full execution: the time for this is not included in per-stage measurement
		cudaStreamSynchronize(my_stream);

		// Stage 2: device to host latency
		start_cycles_d2h = get_cycles();
		for(j = 0; j < NUM_PKTS; j ++) {
			int kernel_inp = (i & 0xff) + j;
			if(B[tid][j] != kernel_inp * kernel_inp) {
				fprintf(stderr, "Kernel output mismatch error\n");
				exit(-1);
			}
		}
		end_cycles_d2h = get_cycles();

		end_cycles = get_cycles();

		tot_cycles_h2d += (end_cycles_h2d - start_cycles_h2d);
		tot_cycles_kernel += (end_cycles_kernel - start_cycles_kernel);
		tot_cycles_d2h += (end_cycles_d2h - start_cycles_d2h);
		tot_cycles += (end_cycles - start_cycles);

		if(rand() % 100 == 0) {
			printf("Thread %d, iter %d | "
				"h2d = %d ns, kernel = %d ns, d2h = %d ns, full = %d ns\n",
				tid, i, 
				(int) ((end_cycles_h2d - start_cycles_h2d) / 2.7),
				(int) ((end_cycles_kernel - start_cycles_kernel) / 2.7),
				(int) ((end_cycles_d2h - start_cycles_d2h) / 2.7),
				(int) ((end_cycles - start_cycles) / 2.7));
		}
		
		err = cudaGetLastError();
		CPE(err != cudaSuccess, "Fail!\n", -1);
	}

	printf("\nFull execution stats:\n");
	printf("\tThread %d: h2d = %d ns, kernel = %d ns, d2h = %d ns, full execution = %d ns\n",
		tid,
		(int) (tot_cycles_h2d / (2.7 * ITERS)),
		(int) (tot_cycles_kernel / (2.7 * ITERS)),
		(int) (tot_cycles_d2h / (2.7 * ITERS)), 
		(int) (tot_cycles / (2.7 * ITERS)));

	long long total_busy_cycles = tot_cycles_h2d + tot_cycles_kernel + tot_cycles_d2h;
	printf("\nSynchronization time = %d ns\n", 
		(int) ((tot_cycles - total_busy_cycles) / (2.7 * ITERS)));

	return 0;
}

int main(void)
{
	int err = cudaSuccess;
	printDeviceProperties();

	// Allocate host vectors in managed memory
	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		err = cudaMallocManaged(&A[thread_i], NUM_PKTS * sizeof(int));
		err = cudaMallocManaged(&B[thread_i], NUM_PKTS * sizeof(int));
		CPE(err != cudaSuccess, "Could not allocate managed memory\n", -1);

		assert(A[thread_i] != NULL);

		// Initialize the managed memory vectors
		for(int j = 0; j < NUM_PKTS; j++)	{
			A[thread_i][j] = thread_i + j;
			B[thread_i][j] = 0;
		}
	}

	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		tid[thread_i] = thread_i;
		pthread_create(&thread[thread_i], NULL, gpu_run, &tid[thread_i]);
	}

	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		pthread_join(thread[thread_i], NULL);
	}	

	// Free allocated managed memory
	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		cudaFree(A[thread_i]);
		cudaFree(B[thread_i]);
	}

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

