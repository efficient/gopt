#include "ipv4.h"

void ipv4_cache_init(uint8_t **ipv4_cache)
{
	int i;

	printf("Initializing ipv4 address cache of size = %lu bytes\n", 
		IPv4_CACHE_CAP * sizeof(uint8_t));

	int sid = shmget(IPv4_CACHE_KEY, IPv4_CACHE_CAP * sizeof(uint8_t), 
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("Could not create ipv4 address cache hash\n");
		exit(-1);
	}

	*ipv4_cache = shmat(sid, 0, 0);

	// Allocate the packets and put them into the hash index randomly
	printf("Putting addresses into ipv4 address cache randomly\n");

	for(i = 0; i < IPv4_CACHE_CAP; i++) {
		(*ipv4_cache)[i] = i;
	}
}
