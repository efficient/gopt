#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <assert.h>

#include "city.h"

#define HASH_INDEX_KEY 1

#define HASH_INDEX_N (64 * 1024 * 1024)		// Number of cuckoo slots
#define HASH_INDEX_N_ ((64 * 1024 * 1024) - 1)

// Number of packets to populate index. Keep this number smaller
// than the number of cuckoo slots so that fewer entries are lost
// due to collision
#define NUM_ENTRIES (16 * 1024 * 1024)	// Number of packets to populate index
#define NUM_ENTRIES_ ((16 * 1024 * 1024) - 1)

struct cuckoo_slot
{
	int key;
	int port;
};

// Cuckoo-specific function prototypes
uint32_t hash(uint32_t u);
void cuckoo_init(int **entries, struct cuckoo_slot** ht_index, int portmask);
