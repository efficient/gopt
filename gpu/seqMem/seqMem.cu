#include "common.h"
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>

cudaStream_t myStream;

__global__ void
seqMem(const long long *log, long long *sum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j, num_iters = LOG_CAP / CUDA_THREADS;
	int iter_base;

	sum[i] = 0;

	for(j = 0; j < num_iters; j ++) {
		iter_base = j * CUDA_THREADS;	
		sum[i] += log[iter_base + i];
	}
}

double cpu_run(long long *log)
{
	int i;
	long long sum = 0;
	
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for(i = 0; i < LOG_CAP; i ++) {
		sum += log[i];
	}

	clock_gettime(CLOCK_REALTIME, &end);

	printf("cpu_run: sum = %lld\n", sum);

	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);
	return time;
}

double gpu_run(long long *d_log)
{
	struct timespec start, end;
	int err = cudaSuccess;
	long long *h_sum, *d_sum, tot_sum = 0;

	int i;

	assert(LOG_CAP % CUDA_THREADS == 0);

	/**< Allocate a sum-buffer on the GPU */
	h_sum = (long long *) malloc(CUDA_THREADS * sizeof(long long));
	err = cudaMalloc((void **) &d_sum, CUDA_THREADS * sizeof(long long));
	CPE(err != cudaSuccess, "Failed to allocate sum buffer on GPU\n", -1);

	/**< Kernel launch */
	int threadsPerBlock = 256;
	int blocksPerGrid = CUDA_THREADS / threadsPerBlock;

	clock_gettime(CLOCK_REALTIME, &start);

	seqMem<<<blocksPerGrid, threadsPerBlock, 0, myStream>>>(d_log, d_sum);
	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch seqMem kernel\n", -1);

	/**< Wait for the kernel to complete */
	cudaStreamSynchronize(myStream);

	clock_gettime(CLOCK_REALTIME, &end);

	/**< Copy back the sum buffer */
	err = cudaMemcpyAsync(h_sum, d_sum, CUDA_THREADS * sizeof(long long),
		cudaMemcpyDeviceToHost, myStream);
	CPE(err != cudaSuccess, "Failed to copy C from device to host\n", -1);

	for(i = 0; i < CUDA_THREADS; i ++) {
		tot_sum += h_sum[i];
	}

	printf("gpu_run: sum = %lld\n", tot_sum);

	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);
	
	return time;
}

int main(int argc, char *argv[])
{
	int err = cudaSuccess;
	int i;
	long long *h_log, *d_log;

	srand(time(NULL));

	printDeviceProperties();

	/** <Initialize a cudaStream for async calls */
	err = cudaStreamCreate(&myStream);
	CPE(err != cudaSuccess, "Failed to create cudaStream\n", -1);

	printf("Creating log of size %lu bytes\n", LOG_CAP * sizeof(long long));
	/** <Initialize hugepage log and copy it to the device: do it once */
#if USE_HUGEPAGE == 1
	int sid = shmget(1, LOG_CAP * sizeof(long long), SHM_HUGETLB | 0666 | IPC_CREAT);
	assert(sid >= 0);
	h_log = (long long *) shmat(sid, 0, 0);
#else
	h_log = (long long *) malloc(LOG_CAP * sizeof(long long));
#endif
	assert(h_log != NULL);

	for(i = 0; i < LOG_CAP; i ++) {
		h_log[i] = i;
	}
	err = cudaMalloc((void **) &d_log, LOG_CAP * sizeof(long long));
	CPE(err != cudaSuccess, "Failed to allocate log on device\n", -1);

	printf("Copying log to device\n");
	err = cudaMemcpy(d_log, h_log, LOG_CAP * sizeof(long long), cudaMemcpyHostToDevice);
	CPE(err != cudaSuccess, "Failed to copy to device memory\n", -1);

	double cpu_time, gpu_time;

	cpu_time = cpu_run(h_log);
	gpu_time = gpu_run(d_log);
	gpu_time = gpu_run(d_log) + gpu_time / 10000000000;
	
	printf("CPU: time = %f, %d GB/s\n", cpu_time,
		(int) ((LOG_CAP * sizeof(long long)) / (cpu_time * 1000000000)));
	printf("GPU: time = %f, %d GB/s\n", gpu_time,
		(int) ((LOG_CAP * sizeof(long long)) / (gpu_time * 1000000000)));
	printf("\n");
	
	// Free device memory
	cudaFree(d_log);

#if USE_HUGEPAGE == 0
	free(h_log);
#endif

	// Reset the device and exit
	err = cudaDeviceReset();
	CPE(err != cudaSuccess, "Failed to de-initialize the device\n", -1);

	printf("Done\n");
	return 0;
}

