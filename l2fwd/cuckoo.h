#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "city.h"

#define SLOTS_PER_BKT 8

#define HASH_INDEX_KEY 1

#define HASH_INDEX_N (128 * 1024 * 1024)		// Number of hash index buckets
#define HASH_INDEX_N_ ((128 * 1024 * 1024) - 1)

#define NUM_ENTRIES (16 * 1024 * 1024)	// Number of packets to populate index
#define NUM_ENTRIES_ ((16 * 1024 * 1024) - 1)

struct cuckoo_slot
{
	int key;
	int value;
};

// Cuckoo-specific function prototypes
uint32_t hash(uint32_t u);
void cuckoo_init(int **entries, struct cuckoo_slot** ht_index);
