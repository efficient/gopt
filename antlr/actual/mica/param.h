#define SLOTS_PER_BKT 8
#define SLOTS_PER_BKT_ 7

#define NUM_PKTS (16 * 1024 * 1024)

#define HT_INDEX_SID 1
#define HT_INDEX_N (8 * 1024 * 1024)		// Number of hash index buckets (size = x 64)
#define HT_INDEX_N_ ((8 * 1024 * 1024) - 1)	// Number of hash index buckets (size = x 64)

#define HT_LOG_SID 2
#define HT_LOG_CAP (16 * 1024 * 1024)			// Number of key-value items in log (size = x 16)
#define HT_LOG_CAP_ ((16 * 1024 * 1024)	- 1)	// Number of key-value items in log (size = x 16)
