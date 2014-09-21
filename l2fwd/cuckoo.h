#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <assert.h>

#include "city.h"
#include "util.h"

#define HASH_INDEX_KEY 2
#define CUCKOO_PORT_MASK 0xf

#define NUM_BKT (8 * 1024 * 1024)		// Number of cuckoo buckets
#define NUM_BKT_ (NUM_BKT - 1)

// Number of packets to populate index. Keep this number smaller
// than the number of cuckoo slots so that fewer entries are lost
// due to collision
#define NUM_MAC (16 * 1024 * 1024)
#define NUM_MAC_ (NUM_MAC - 1)

struct cuckoo_slot
{
	uint32_t mac;
	int port;
};

struct cuckoo_bucket
{
	struct cuckoo_slot slot[8];
};

void cuckoo_init(uint32_t **mac_addrs, 
	struct cuckoo_bucket** ht_index, int portmask);
