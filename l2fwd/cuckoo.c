#include "cuckoo.h"

/**< Don't want to include rte header for RTE_MAX_ETHPORTS */
#define CUCKOO_MAX_ETHPORTS 16
#define CUCKOO_ISSET(a, i) (a & (1 << i))

void cuckoo_init(uint32_t **mac_addrs, 
	struct cuckoo_bucket **ht_index, int portmask)
{
	int i, overwrites = 0;
	uint32_t rand_1, rand_2;
	uint32_t mac_32;

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
	*mac_addrs = malloc(NUM_MAC * sizeof(uint32_t));

	for(i = 0; i < NUM_MAC; i++) {
		rand_1 = (uint32_t) rand();
		rand_2 = (uint32_t) rand();
		mac_32 = rand_1 ^ (rand_2 << 1);

		(*mac_addrs)[i] = mac_32;

		// Choose one of the two candidate buckets randomly
		bkt = CityHash32((char *) &mac_32, 4) & NUM_BKT_;
		if(rand() % 2 != 0) {
			bkt = CityHash32((char *) &bkt, 4) & NUM_BKT_;
		}

		int success = 0;
		for(slot_i = 0; slot_i < 8; slot_i ++) {
			/**< Find an empty slot */
			if((*ht_index)[bkt].slot[slot_i].mac == 0) {
				(*ht_index)[bkt].slot[slot_i].mac = mac_32;
				int port = port_arr[rand() % num_active_ports];
				(*ht_index)[bkt].slot[slot_i].port = port;
				success = 1;
				break;
			}
		}

		if(success == 0) {
			/**< Failed to find an empty slot */
			slot_i = rand() % 8;
			(*ht_index)[bkt].slot[slot_i].mac = mac_32;
			int port = port_arr[rand() % num_active_ports];
			(*ht_index)[bkt].slot[slot_i].port = port;
			success = 1;
	
			overwrites ++;
		}
	}

	printf("Percentage of entries overwritten = %f\n", (double) overwrites / NUM_MAC);
}

/**< Return the mapped port for a 32-bit request MAC */
int cuckoo_lookup(uint32_t req, struct cuckoo_bucket *ht_index)
{
	int bkt_1, bkt_2, fwd_port = -1;
	int j;
	
	uint32_t dst_mac = req;
	bkt_1 = CityHash32((char *) &dst_mac, 4) & NUM_BKT_;

	for(j = 0; j < 8; j ++) {
		uint32_t slot_mac = ht_index[bkt_1].slot[j].mac;
		int slot_port = ht_index[bkt_1].slot[j].port;

		if(slot_mac == dst_mac) {
			assert(slot_port != -1);
			fwd_port = slot_port;
			break;
		}
	}

	if(fwd_port == -1) {
		bkt_2 = CityHash32((char *) &bkt_1, 4) & NUM_BKT_;

		for(j = 0; j < 8; j ++) {
			uint32_t slot_mac = ht_index[bkt_2].slot[j].mac;
			int slot_port = ht_index[bkt_2].slot[j].port;

			if(slot_mac == dst_mac) {
				fwd_port = slot_port;
				break;
			}
		}
	}

	return fwd_port;
}
