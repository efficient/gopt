#define INCLUDE_COPY_TIME 1

#define ITERS 100			/** < Number of measurements to average on */

#define MAX_PKTS (1024 * 32)

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}

void printDeviceProperties();
