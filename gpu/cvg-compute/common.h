#define INCLUDE_COPY_TIME 1

#define DEPTH 10	/**< Number of CityHash computations*/

#define NUM_PKTS 8192

#define GPU_MAX_PKTS (32768 * 128) /**< Max packets to use for GPU */
#define GPU_ITERS 10 /**< Iters to average GPU measurements */

#define CPU_MAX_THREADS 8
#define CPU_ITERS 1000 /**< Iters to average CPU measurements */

/**< The speed of dependent CityHashes can be increased by a loop
  *  interchange. However, doing so would be cheating bc we want to
  *  emulate expensive computation via a series of dependent CityHashes */
#define LOOP_INTERCHANGE 0

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}

void red_printf(const char *format, ...);

struct thread_info {
	int tid;
};
