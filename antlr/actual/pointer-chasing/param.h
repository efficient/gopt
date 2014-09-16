#define DEPTH 100
#define NUM_PKTS (1 * 1024 * 1024)

#define LOG_SID 1

/** < 512 MB: DRAM */
//#define LOG_CAP (128 * 1024 * 1024)		// Number of ints in the log
//#define LOG_CAP_ (LOG_CAP - 1)

/** < 16 MB: L3 cache */
//#define LOG_CAP (4 * 1024 * 1024)			// Number of ints in the log
//#define LOG_CAP_ (LOG_CAP - 1)

/** < 256K: L2 cache */
#define LOG_CAP (64 * 1024)					// Number of ints in the log
#define LOG_CAP_ (LOG_CAP - 1)
