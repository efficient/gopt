#define DEPTH 20
#define NUM_PKTS (1 * 1024 * 1024)

#define LOG_SID 1
#define LOG_CAP (128 * 1024 * 1024)		// Number of ints in the log
#define LOG_CAP_ ((128 * 1024 * 1024) - 1)

// Make the log fit inside L3
//#define LOG_CAP (512 * 1024)		// Number of ints in the log
//#define LOG_CAP_ ((512 * 1024) - 1)
