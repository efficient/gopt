#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>

#define LOG_CAP (192 * 1024 * 1024)		// 256 M log
#define LOG_CAP_ ((192 * 1024 * 1024) - 1)	// 256 M log

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}

double cpu_run(long long *log)
{
	int i;
	long long sum = 0;

	/**< Touch one cacheline at a time */
	int step = (int) (64 / sizeof(long long));
	
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for(i = 0; i < LOG_CAP; i += step) {
		sum += log[i];
	}

	clock_gettime(CLOCK_REALTIME, &end);

	printf("cpu_run: sum = %lld\n", sum);

	double time = (double) (end.tv_nsec - start.tv_nsec) / 1000000000 + 
		(end.tv_sec - start.tv_sec);
	return time;
}

int main(int argc, char *argv[])
{
	int i;
	long long *log;

	srand(time(NULL));

	unsigned long log_bytes = LOG_CAP * sizeof(long long);

	printf("Creating log of size %lu bytes\n", log_bytes);
	log = (long long *) malloc(log_bytes);
	assert(log != NULL);

	for(i = 0; i < LOG_CAP; i ++) {
		log[i] = i;
	}

	double cpu_time;

	while(1) {
		cpu_time = cpu_run(log);
		
		printf("CPU: time = %f, %.2f GB/s\n", cpu_time,
			(log_bytes) / (cpu_time * 1000000000));
	}
	
	free(log);

	return 0;
}

