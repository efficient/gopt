#include "cuckoo.h"

/**< Don't want to include rte header for RTE_MAX_ETHPORTS */
#define CUCKOO_MAX_ETHPORTS 16
#define CUCKOO_ISSET(a, i) (a & (1 << i))

void cuckoo_init(ULL **mac_addrs, 
	struct cuckoo_bucket **ht_index, int portmask)
{
	int i, overwrites = 0;
	LL rand_1, rand_2;
	ULL mac_48;
	uint8_t __mac_48[6];

	int bkt, slot_i;

	int num_active_ports = bitcount(portmask);
	int *port_arr = get_active_bits(portmask);

	printf("Initializing cuckoo index of size = %lu bytes\n", 
		NUM_BKT * sizeof(struct cuckoo_bucket));

	int sid = shmget(HASH_INDEX_KEY, NUM_BKT * sizeof(struct cuckoo_bucket), 
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("Could not create cuckoo hash index\n");
		exit(-1);
	}

	*ht_index = shmat(sid, 0, 0);
	memset((char *) *ht_index, 0, NUM_BKT * sizeof(struct cuckoo_bucket));

	srand(2);

	/**< Allocate the packets and put them into the hash index randomly */
	printf("Putting active ports into hash index randomly\n");
	*mac_addrs = malloc(NUM_MAC * sizeof(ULL));

	for(i = 0; i < NUM_MAC; i++) {
		rand_1 = (LL) lrand48();
		rand_2 = (LL) lrand48();
		mac_48 = (ULL) (rand_1 ^ (rand_2 << 16));
		assert(mac_48 <= 0xffffffffffffL && mac_48 > 0);

		/**< Store the mac so that the client can use it for probes later */
		(*mac_addrs)[i] = mac_48;

		/**< Put the mac into a byte-array for hash computation */
		set_mac(__mac_48, mac_48);

		/**< Choose one of the two candidate buckets randomly */
		bkt = CityHash32((char *) __mac_48, 6) & NUM_BKT_;
		if(rand() % 2 != 0) {
			bkt = CityHash32((char *) &bkt, 4) & NUM_BKT_;
		}

		int success = 0;
		for(slot_i = 0; slot_i < 8; slot_i ++) {
			/**< Find an empty slot */
			if(SLOT_TO_MAC((*ht_index)[bkt].slot[slot_i]) == 0) {
				ULL port = port_arr[rand() % num_active_ports];
				(*ht_index)[bkt].slot[slot_i] = (port << 48) + mac_48;
				success = 1;
				break;
			}
		}

		if(success == 0) {
			/**< Failed to find an empty slot */
			slot_i = rand() % 8;
			ULL port = port_arr[rand() % num_active_ports];
			(*ht_index)[bkt].slot[slot_i] = (port << 48) + mac_48;
			success = 1;
	
			overwrites ++;
		}
	}

	printf("Percentage of entries overwritten = %f\n", (double) overwrites / NUM_MAC);
}
