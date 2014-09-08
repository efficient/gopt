#include "common.h"
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#define CPU_GHZ 2.7		// xia-router2
#define COMPUTE 20

__global__ void
vectorAdd(int *pkts, int num_pkts)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_pkts) {
		int j;
		for(j = 0; j < COMPUTE; j ++) {
			pkts[i] = (pkts[i] + 1) * pkts[i];
		}
	}
}

double cpu_run(int *pkts, int num_pkts)
{
	int i;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	long long c1 = get_cycles();

	for(i = 0; i < num_pkts; i += 1) {
		int j;
		for(j = 0; j < COMPUTE; j ++) {
			pkts[i] = (pkts[i] + 1) * pkts[i];
		}
	}

	long long c2 = get_cycles();
	printf("Nanoseconds per packet = %f\n", 
		(double) (c2 - c1 - 90) / (CPU_GHZ * num_pkts));

	clock_gettime(CLOCK_REALTIME, &end);
	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);
	return time;
}

double gpu_run(int *h_pkts, int num_pkts)
{
	struct timespec start, end;
	int *d_pkts = NULL;
	int err = cudaSuccess;

	err = cudaMalloc((void **) &d_pkts, num_pkts * sizeof(int));
	CPE(err != cudaSuccess, "Failed to cudaMalloc\n", -1);

	// Start the clock
	clock_gettime(CLOCK_REALTIME, &start);
	
	err = cudaMemcpy(d_pkts, h_pkts, num_pkts * sizeof(int), cudaMemcpyHostToDevice);
	CPE(err != cudaSuccess, "Failed to copy to device memory\n", -1);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (num_pkts + threadsPerBlock - 1) / threadsPerBlock;

	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_pkts, num_pkts);
	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch vectorAdd kernel\n", -1);

	// Copy back the result
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

	srand(time(NULL));

	printDeviceProperties();

	for(int num_pkts = 8; num_pkts < 8 * 1024; num_pkts += 8) {

		int *h_pkts_cpu = (int *) malloc(num_pkts * sizeof(int));
		int *h_pkts_gpu = (int *) malloc(num_pkts * sizeof(int));
		double cpu_time, gpu_time;

		// Verify that allocations succeeded
		if (h_pkts_cpu == NULL || h_pkts_gpu == NULL) {
			fprintf(stderr, "Failed to allocate host mem!\n");
			exit(-1);
		}
		
		// Initialize packets
		for(i = 0; i < num_pkts; i ++) {
			h_pkts_cpu[i] = rand();
			h_pkts_gpu[i] = h_pkts_cpu[i];
		}
	
		cpu_time = cpu_run(h_pkts_cpu, num_pkts);
		gpu_time = gpu_run(h_pkts_gpu, num_pkts);
	
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

