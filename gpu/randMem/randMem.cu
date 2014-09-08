#include "common.h"
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>

__global__ void
randMem(int *pkts, const int *log, int num_pkts)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_pkts) {
		int j;
		for(j = 0; j < DEPTH; j ++) {
			pkts[i] = log[pkts[i]];
		}
	}
}

double cpu_run(int *pkts, int *log, int num_pkts)
{
	int i;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

#if USE_INTERLEAVING == 1
	for(i = 0; i < num_pkts; i += 8) {
		int j;
		for(j = 0; j < DEPTH; j ++) {
			pkts[i] = log[pkts[i]];
			pkts[i + 1] = log[pkts[i + 1]];
			pkts[i + 2] = log[pkts[i + 2]];
			pkts[i + 3] = log[pkts[i + 3]];
			pkts[i + 4] = log[pkts[i + 4]];
			pkts[i + 5] = log[pkts[i + 5]];
			pkts[i + 6] = log[pkts[i + 6]];
			pkts[i + 7] = log[pkts[i + 7]];
		}
	}
#else
	for(i = 0; i < num_pkts; i += 1) {
		int j;
		for(j = 0; j < DEPTH; j ++) {
			pkts[i] = log[pkts[i]];
		}
	}
#endif
	

	clock_gettime(CLOCK_REALTIME, &end);
	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);
	return time;
}

double gpu_run(int *h_pkts, int *d_log, int num_pkts)
{
	struct timespec start, end;
	int *d_pkts = NULL;
	int err = cudaSuccess;

	err = cudaMalloc((void **) &d_pkts, num_pkts * sizeof(int));
	CPE(err != cudaSuccess, "Failed to allocate packet buffer on device\n", -1);

	// Start the clock
	clock_gettime(CLOCK_REALTIME, &start);
	err = cudaMemcpy(d_pkts, h_pkts, num_pkts * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (num_pkts + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	randMem<<<blocksPerGrid, threadsPerBlock>>>(d_pkts, d_log, num_pkts);
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch randMem kernel\n", -1);

	// Copy back the result
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_pkts, d_pkts, num_pkts * sizeof(int), cudaMemcpyDeviceToHost);
	CPE(err != cudaSuccess, "Failed to copy C from device to host\n", -1);

	clock_gettime(CLOCK_REALTIME, &end);
	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);

	// Free device global memory
	err = cudaFree(d_pkts);
	CPE(err != cudaSuccess, "Failed to cudaFree\n", -1);

	return time;
}

int main(int argc, char *argv[])
{
	int err = cudaSuccess;
	int i;
	int *h_log, *d_log;

	srand(time(NULL));

	printDeviceProperties();

	// Initialize hugepage log once for all values of num_pkts
#if USE_HUGEPAGE == 1
	int sid = shmget(1, LOG_CAP * sizeof(int), SHM_HUGETLB | 0666 | IPC_CREAT);
	assert(sid >= 0);
	h_log = (int *) shmat(sid, 0, 0);
	assert(h_log != NULL);
#else
	h_log = (int *) malloc(LOG_CAP * sizeof(int));
	assert(h_log != NULL);
#endif

	// Initialize the log and copy it to the device once
	for(i = 0; i < LOG_CAP; i ++) {
		h_log[i] = rand() % LOG_CAP;
	}
	err = cudaMalloc((void **) &d_log, LOG_CAP * sizeof(int));
	CPE(err != cudaSuccess, "Failed to allocate log on device\n", -1);

	err = cudaMemcpy(d_log, h_log, LOG_CAP * sizeof(int), cudaMemcpyHostToDevice);
	CPE(err != cudaSuccess, "Failed to copy to device memory\n", -1);

	// Test for different batch sizes
	for(int num_pkts = 8; num_pkts < 8 * 1024; num_pkts += 8) {

		int *h_pkts_cpu = (int *) malloc(num_pkts * sizeof(int));
		int *h_pkts_gpu = (int *) malloc(num_pkts * sizeof(int));
		double cpu_time, gpu_time;

		// Verify that allocations succeeded
		if (h_pkts_cpu == NULL || h_pkts_gpu == NULL || h_log == NULL) {
			fprintf(stderr, "Failed to allocate host mem!\n");
			exit(-1);
		}
		
		// Initialize packets
		for(i = 0; i < num_pkts; i ++) {
			h_pkts_cpu[i] = rand() & LOG_CAP_;
			h_pkts_gpu[i] = h_pkts_cpu[i];
		}
	
		cpu_time = cpu_run(h_pkts_cpu, h_log, num_pkts);
		gpu_time = gpu_run(h_pkts_gpu, d_log, num_pkts);
	
		// Verify that the result vector is correct
		for(int i = 0; i < num_pkts; i ++) {
			if (h_pkts_cpu[i] != h_pkts_gpu[i]) {
				fprintf(stderr, "Result verification failed at element %d!\n", i);
				fprintf(stderr, "CPU %d, GPU %d\n", h_pkts_cpu[i], h_pkts_gpu[i]);
				exit(-1);
			}
		}
	
		printf("Test PASSED for num_pkts = %d\n", num_pkts);

		// Emit the results to stderr. Use only space for delimiting
		fprintf(stderr, "Batch size  %d CPU %f GPU %f CPU/GPU %f\n",
			num_pkts, cpu_time, gpu_time, cpu_time / gpu_time);
	
		// Free host memory
		free(h_pkts_cpu);
		free(h_pkts_gpu);
	}

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

