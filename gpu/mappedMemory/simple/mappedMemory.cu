#include "common.h"

int volatile *h_A, *h_B;
int volatile *d_A, *d_B;
int volatile *h_flag, *d_flag;
int NUM_PKTS = -1;			/** < Passed as a command-line flag */

pthread_t cpu_thread;		/** < CPU thread that talks to the GPU */

__global__ void
vectorAdd(volatile int *A, volatile int *B, volatile int *flag, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int iter = 0;
	
	if(i < N) {
		/** < Wait for CPU flag a finite number of times */
		for(iter = 0; iter < ITERS; iter ++) {
			while(flag[0] == iter - 1) {
				// Do nothing
			}

			/** < Don't touch A[i] twice */
			B[i] = A[i] + 1;
		}
	}
}

void *gpu_run(void *ptr)
{
	/** < We can only get full execution measurements */
	struct timespec start, end;
	double diff[ITERS];
	double tot;

	int iter = 0, j = 0;

	for(iter = 0; iter < ITERS; iter ++) {

		/** < Set B to zero: the GPU will make it non-zero */
		memset((char *) h_B, 0, NUM_PKTS * sizeof(int));
	
		/** < Start a timer */
		clock_gettime(CLOCK_REALTIME, &start);
		
		/** < Write input data into A */
		for(j = 0; j < NUM_PKTS; j ++) {
			h_A[j] = iter + 1;		// Always > 0
			assert(h_A[j] != 0);
		}

		/** < Raise a flag for the GPU */
		h_flag[0] = iter;

		/** < Wait till the GPU makes all of h_B non-zero */
		waitForNonZero(h_B, NUM_PKTS);

		for(j = 0; j < NUM_PKTS; j ++) {
			if(h_B[j] != h_A[j] + 1) {
				fprintf(stderr, "Kernel output mismatch error\n");
				exit(-1);
			}
		}

		/** < Stop timer */
		clock_gettime(CLOCK_REALTIME, &end);

		diff[iter] = get_timespec_us(start, end);
		tot += diff[iter];

		printf("\tIter %d: %.2f us\n", iter, diff[iter]);
	}

	/** < Sort the times for percentile */
	qsort(diff, ITERS, sizeof(double), cmpfunc);
	red_printf("Average %.2f 5th %.2f 95th %.2f\n",
		tot / ITERS, diff[0], diff[(ITERS * 95) / 100]);

	return 0;
}

int main(int argc, char **argv)
{
	/** < NUM_PKTS is passed as a command-line flag */
	assert(argc == 2);
	NUM_PKTS = atoi(argv[1]);

	int threadsPerBlock = NUM_PKTS;
	int blocksPerGrid = (NUM_PKTS + threadsPerBlock - 1) / threadsPerBlock;

	int err = cudaSuccess;
	printDeviceProperties();

	/** < Enable mapped-memory on device */
	cudaSetDeviceFlags(cudaDeviceMapHost);

	/** < Allocate the mapped-memory regions */
	err = cudaHostAlloc(&h_A, NUM_PKTS * sizeof(int), cudaHostAllocMapped);
	err = cudaHostAlloc(&h_B, NUM_PKTS * sizeof(int), cudaHostAllocMapped);
	err = cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped);
	CPE(err != cudaSuccess, "Could not allocate mapped memory\n");

	/** Get device pointers for mapped memory */
	err = cudaHostGetDevicePointer((void **) &d_A, (void *) h_A, 0);
	err = cudaHostGetDevicePointer((void **) &d_B, (void *) h_B, 0);
	err = cudaHostGetDevicePointer((void **) &d_flag, (void *) h_flag, 0);
	CPE(err != cudaSuccess, "Could not get device pointer for mapped memory\n");

	/** < Init. mapped memory: For iteration #i, i >= 0, the flag is i */
	h_flag[0] = -1;
	for(int i = 0; i < NUM_PKTS; i++) {
		h_A[i] = 0;
		h_B[i] = 0;
	}

	/** < Launch a CPU thread that talks to the GPU */
	pthread_create(&cpu_thread, NULL, gpu_run, NULL);

	/** < Launch the kernel once */
	red_printf("Launching CUDA kernel\n");

	cudaStream_t my_stream;
	err = cudaStreamCreate(&my_stream);
	CPE(err != cudaSuccess, "Failed to create cudaStream\n");

	vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_A, 
		d_B, d_flag, NUM_PKTS);
	cudaStreamQuery(my_stream);

	printf("Waiting for CPU thread to finish\n");
	pthread_join(cpu_thread, NULL);

	/** < Free allocated mapped memory */
	cudaFreeHost((void *) h_A);
	cudaFreeHost((void *) h_B);
	cudaFreeHost((void *) h_flag);

	/** < Reset the device and exit */
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n");

	printf("Done\n");
	return 0;
}

