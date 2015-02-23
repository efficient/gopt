#define INCLUDE_COPY_TIME 1

#define ITERS 10	/**< Number of measurements to average on */
#define DEPTH 2		/**< Length of pointer chain */

#define MAX_THREADS 8
#define NUM_PKTS 8192	/**< Number of packets to use for CPU */
#define MAX_PKTS (32768 * 128)	/**< Max packets to use for GPU */

#define LOG_KEY 1
#define LOG_CAP (256 * 1024 * 1024)		/**< 1 GB */
#define LOG_CAP_ ((256 * 1024 * 1024) - 1)

#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}

void red_printf(const char *format, ...);

struct thread_info {
	int tid;
	int *log;
};
