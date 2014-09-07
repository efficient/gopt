// Example code for CUDA streams

#include "common.h"

#define NUM_THREADS 4

int *h_A[NUM_THREADS];
int tid[NUM_THREADS];
pthread_t thread[NUM_THREADS];

__global__ void
vectorAdd(int *A, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		A[i] *= A[i];
	}
}

void *gpu_run(void *ptr)
{
	int tid = *((int *) ptr);
	printf("My tid = %d\n", tid);

	cudaStream_t my_stream;
		
	int *d_A = NULL;
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

	err = cudaMalloc((void **) &d_A, NUM_PKTS * sizeof(int));
	CPE(err != cudaSuccess, "Failed to cudaMalloc\n", -1);

	// Measure host-to-device memcpy latency
	for(i = 0; i < ITERS; i ++) {
		start_cycles = get_cycles();
		
		// Initialize h_A
		for(j = 0; j < NUM_PKTS; j ++) {
			h_A[tid][j] = (i & 0xff) + j;
		}

		// Stage 1: host to device memcpy
		start_cycles_h2d = get_cycles();
		cudaMemcpyAsync(d_A, h_A[tid], NUM_PKTS * sizeof(int), cudaMemcpyHostToDevice, my_stream);
		end_cycles_h2d = get_cycles();

		// Stage 2: kernel execution
		start_cycles_kernel = get_cycles();
		vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_A, NUM_PKTS);
		end_cycles_kernel = get_cycles();

		// Stage 2: device to host memcpy
		start_cycles_d2h = get_cycles();
		cudaMemcpyAsync(h_A[tid], d_A, NUM_PKTS * sizeof(int), cudaMemcpyDeviceToHost, my_stream);
		end_cycles_d2h = get_cycles();

		// Complete full execution: the time for this is not included in per-stage measurement
		cudaStreamSynchronize(my_stream);

		// Verify kernel's result
		for(j = 0; j < NUM_PKTS; j ++) {
			int kernel_inp = (i & 0xff) + j;
			if(h_A[tid][j] != kernel_inp * kernel_inp) {
				fprintf(stderr, "Kernel output mismatch error\n");
				exit(-1);
			}
		}

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

	cudaFree(d_A);

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

	// Allocate host vectors in pinned memory
	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		err = cudaMallocHost((void **) &h_A[thread_i], NUM_PKTS * sizeof(int));
		CPE(err != cudaSuccess, "Could not allocate pinned memory\n", -1);

		// Verify that allocations succeeded
		if (h_A[thread_i] == NULL) {
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit(EXIT_FAILURE);
		}

		// Initialize the host input vectors
		for(int j = 0; j < NUM_PKTS; j++)	{
			h_A[thread_i][j] = thread_i + j;
		}
	}

	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		tid[thread_i] = thread_i;
		pthread_create(&thread[thread_i], NULL, gpu_run, &tid[thread_i]);
	}

	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		pthread_join(thread[thread_i], NULL);
	}	

	// Free pinned host memory
	for(int thread_i = 0; thread_i < NUM_THREADS; thread_i ++) {
		cudaFreeHost(h_A[thread_i]);
	}

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

