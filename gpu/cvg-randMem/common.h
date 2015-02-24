#define INCLUDE_COPY_TIME 1

#define DEPTH 1	/**< Length of pointer chains */

#define GPU_MAX_PKTS (32768 * 128) /**< Max packets to use for GPU */
#define GPU_ITERS 10 /**< Iters to average GPU measurements */

#define CPU_MAX_THREADS 8
#define CPU_NUM_STREAMS 8192 /**< Number of concurrent streams for CPU */
#define CPU_ITERS 1000 /**< Iters to average CPU measurements */

#define LOG_KEY 1
#define LOG_CAP (256 * 1024 * 1024)		/**< 1 GB */
#define LOG_CAP_ ((256 * 1024 * 1024) - 1)

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}

void red_printf(const char *format, ...);
void init_ht_log(int *log, int n);

struct thread_info {
	int tid;
	int *log;
};
