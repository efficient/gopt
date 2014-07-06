#define SLOTS_PER_BKT 16

#define BATCH_SIZE 8
#define BATCH_SIZE_ 7

#define DEPTH 2
#define NUM_PKTS (16 * 1024 * 1024)

#define CACHE_SID 1
#define NUM_BS (8 * 1024 * 1024)		// Number of cache buckets (avoiding BKTS)
#define NUM_BS_ ((8 * 1024 * 1024) - 1)

// Make the cache fit inside L3
//#define NUM_BS (131072)		// Number of cache buckets (avoiding BKTS)
//#define NUM_BS_ (131071)
