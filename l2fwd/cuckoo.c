#include "cuckoo.h"

// Don't want to include rte header
#define CUCKOO_MAX_ETHPORTS 16
#define CUCKOO_ISSET(a, i) (a & (1 << i))

uint32_t hash(uint32_t u)
{
	return CityHash32((char *) &u, 4);
}

// Count the number of 1-bits in n
int cuckoo_bitcount(int n)
{
	int count = 0;
	while(n > 0) {
		count ++;
		n = n & (n - 1);
	}
	return count;
}

// Returns an array containing the port numbers of all ports that are active
int *cuckoo_get_active_ports(int portmask)
{
	int num_active_ports = cuckoo_bitcount(portmask);
	int *active_ports = (int *) malloc(num_active_ports * sizeof(int));
	int pos = 0, i;
	for(i = 0; i < CUCKOO_MAX_ETHPORTS; i++) {
		if(CUCKOO_ISSET(portmask, i)) {
			active_ports[pos] = i;
			pos ++;
		}
	}
	assert(pos == num_active_ports);
	return active_ports;
}

void cuckoo_init(int **entries, struct cuckoo_slot **ht_index, int portmask)
{
	int i, overwrites = 0;

	int num_active_ports = cuckoo_bitcount(portmask);
	int *port_arr = cuckoo_get_active_ports(portmask);

	printf("Initializing cuckoo index of size = %lu bytes\n", 
		HASH_INDEX_N * sizeof(struct cuckoo_slot));

	int sid = shmget(HASH_INDEX_KEY, HASH_INDEX_N * sizeof(struct cuckoo_slot), 
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("Could not create cuckoo hash index\n");
		exit(-1);
	}

	*ht_index = shmat(sid, 0, 0);
	memset((char *) *ht_index, 0, HASH_INDEX_N_ * sizeof(struct cuckoo_slot));

	srand(2);

	// Allocate the packets and put them into the hash index randomly
	printf("Putting active ports into hash index randomly\n");
	*entries = malloc(NUM_ENTRIES * sizeof(int));

	for(i = 0; i < NUM_ENTRIES; i++) {
		int K = rand();
		(*entries)[i] = K;
		
		// With 1/2 probability, put into 1st bucket
		int hash_bucket_i = 0;
		
		// The 2nd hash function for key K is CITYHASH(K + 1)
		if(rand() % 2 == 0) {
			hash_bucket_i = hash(K) & HASH_INDEX_N_;
		} else {
			hash_bucket_i = hash(K + 1) & HASH_INDEX_N_;
		}

		if((*ht_index)[hash_bucket_i].key != 0) {
			overwrites ++;
		}

		// The value for key K is a random, enabled, port 
		(*ht_index)[hash_bucket_i].key = K;
		(*ht_index)[hash_bucket_i].port = port_arr[rand() % num_active_ports];
	}

	printf("Percentage of entries overwritten = %f\n", (double) overwrites / NUM_ENTRIES);
}
