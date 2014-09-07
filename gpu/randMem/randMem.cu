#include "common.h"
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>

__global__ void
vectorAdd(int *pkts, const int *log, int num_pkts)
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

double gpu_run(int *h_pkts, int *h_log, int num_pkts)
{
	struct timespec start, end;
	int *d_pkts = NULL, *d_log = NULL;
	int err = cudaSuccess;

	err = cudaMalloc((void **) &d_pkts, num_pkts * sizeof(int));
	err = cudaMalloc((void **) &d_log, LOG_CAP * sizeof(int));
	CPE(err != cudaSuccess, "Failed to cudaMalloc\n", -1);

	err = cudaMemcpy(d_pkts, h_pkts, num_pkts * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_log, h_log, LOG_CAP * sizeof(int), cudaMemcpyHostToDevice);
	CPE(err != cudaSuccess, "Failed to copy to device memory\n", -1);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (num_pkts + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	// Start the clock
	clock_gettime(CLOCK_REALTIME, &start);

	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_pkts, d_log, num_pkts);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_REALTIME, &end);
	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);

	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch vectorAdd kernel\n", -1);

	// Copy back the result
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_pkts, d_pkts, num_pkts * sizeof(int), cudaMemcpyDeviceToHost);
	CPE(err != cudaSuccess, "Failed to copy C from device to host\n", -1);

	// Free device global memory
	err = cudaFree(d_pkts);
	err = cudaFree(d_log);
	CPE(err != cudaSuccess, "Failed to cudaFree\n", -1);

	return time;
}

int main(int argc, char *argv[])
{
	int err = cudaSuccess;
	int i;

	srand(time(NULL));

	printDeviceProperties();

	// Initialize hugepage log once for all values of num_pkts
#if USE_HUGEPAGE == 1
	int sid = shmget(1, LOG_CAP * sizeof(int), SHM_HUGETLB | 0666 | IPC_CREAT);
	assert(sid >= 0);
	int *h_log = (int *) shmat(sid, 0, 0);
	assert(h_log != NULL);
#else
	int *h_log = (int *) malloc(LOG_CAP * sizeof(int));
	assert(h_log != NULL);
#endif

	for(i = 0; i < LOG_CAP; i ++) {
		h_log[i] = rand() % LOG_CAP;
	}

	for(int num_pkts = 8; num_pkts < 1024; num_pkts += 8) {

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
		gpu_time = gpu_run(h_pkts_gpu, h_log, num_pkts);
	
		// Verify that the result vector is correct
		for(int i = 0; i < num_pkts; i ++) {
			if (h_pkts_cpu[i] != h_pkts_gpu[i]) {
				fprintf(stderr, "Result verification failed at element %d!\n", i);
				fprintf(stderr, "CPU %d, GPU %d\n", h_pkts_cpu[i], h_pkts_gpu[i]);
				exit(-1);
			}
		}
	
		printf("Test PASSED\n");
		printf("Batch size = %d, CPU: %f, GPU: %f, CPU/GPU: %f\n",
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

