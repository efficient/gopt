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

struct cuckoo_bucket
{
	/**< Slot: bytes 0:1 = port | bytes 2:7 = mac */
	ULL slot[8];
};

// These macros should be safe for use with the ANTLR code
#define SLOT_TO_MAC(s) (s & ((1L << 48) - 1))
#define SLOT_TO_PORT(s) (s >> 48)

// Cuckoo-specific function prototypes
uint32_t hash(uint32_t u);
void cuckoo_init(ULL **mac_addrs, 
	struct cuckoo_bucket** ht_index, int portmask);

/**< Return the mapped port for a MAC address */
int cuckoo_lookup(uint8_t *dst_mac_ptr, struct cuckoo_bucket *ht_index);
