#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "fpp.h"
#include "param.h"
#include "city.h"

#define foreach(i, n) for(i = 0; i < n; i ++)

// Compute an expensive hash using multiple applications of cityhash
uint32_t cityhash(uint32_t u)
{
	uint32_t ret = u, i;
	for(i = 0; i < DEPTH; i ++) {
		ret = CityHash32((char *) &ret, 4);
	}
	return ret;
}

struct cuckoo_slot
{
	uint32_t key;
	uint32_t value;
};
struct cuckoo_slot *hash_index;


// Each packet contains a random integer. The memory address accessed
// by the packet is determined by an expensive hash of the integer.
uint32_t *pkts;

int sum = 0;
int succ_1 = 0;
int succ_2 = 0;
int fail = 0;

// batch_index must be declared outside process_pkts_in_batch
int batch_index = 0;

void process_pkts_in_batch(uint32_t *pkt_lo) 
{
	int i;
	uint32_t K[BATCH_SIZE], S1[BATCH_SIZE], S2[BATCH_SIZE];
	uint32_t done_mask = 0;

	// Stage 1: issue prefetches for 1st bucket
	for(i = 0; i < BATCH_SIZE; i ++) {
		K[i] = pkt_lo[i];
		S1[i] = cityhash(K[i]) % HASH_INDEX_N;
		__builtin_prefetch(&hash_index[S1[i]], 0, 0);
	}

	// Stage 2: access 1st bucket, issue 2nd prefetch
	for(i = 0; i < BATCH_SIZE; i ++) {
		if(hash_index[S1[i]].key == K[i]) {
			sum += hash_index[S1[i]].value;
			succ_1 ++;
			done_mask = FPP_SET(done_mask, i);
		} else{
			S2[i] = cityhash(K[i] + 1) % HASH_INDEX_N;
			__builtin_prefetch(&hash_index[S2[i]], 0, 0);
		}
	}

	// Stage 3: access 2nd bucket
	for(i = 0; i < BATCH_SIZE; i ++) {
		if(!FPP_ISSET(done_mask, i)) {
			if(hash_index[S2[i]].key == K[i]) {
				sum += hash_index[S2[i]].value;
				succ_2 ++;
			} else {
				fail ++;
			}
		}
	}
}

int main(int argc, char **argv)
{
	int i, j;

	// Allocate a large memory area
	fprintf(stderr, "Size of hash index = %lu\n", HASH_INDEX_N * sizeof(struct cuckoo_slot));

	int sid = shmget(HASH_INDEX_SID, HASH_INDEX_N * sizeof(struct cuckoo_slot), 
		IPC_CREAT | 0666 | SHM_HUGETLB);
	if(sid < 0) {
		fprintf(stderr, "Could not create cuckoo hash index\n");
		exit(-1);
	}
	hash_index = shmat(sid, 0, 0);

	// Allocate the packets and put them into the hash index randomly
	printf("Putting packets into hash index randomly\n");
	pkts = (uint32_t *) malloc(NUM_PKTS * sizeof(int));
	for(i = 0; i < NUM_PKTS; i++) {
		uint32_t K = (uint32_t) rand();
		pkts[i] = K;
		
		// With 1/2 probability, put into 1st bucket
		uint32_t hash_bucket_i = 0;
		
		// The 2nd hash function for key K is CITYHASH(K + 1)
		if(rand() % 2 == 0) {
			hash_bucket_i = cityhash(K) % HASH_INDEX_N;
		} else {
			hash_bucket_i = cityhash(K + 1) % HASH_INDEX_N;
		}

		// The value for key K is K + i
		hash_index[hash_bucket_i].key = K;
		hash_index[hash_bucket_i].value = K + i;
	}

	printf("Starting lookups\n");
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for(i = 0; i < NUM_PKTS; i += BATCH_SIZE) {
		process_pkts_in_batch(&pkts[i]);
	}

	clock_gettime(CLOCK_REALTIME, &end);
	printf("Time = %f sum = %d, succ_1 = %d, succ_2 = %d, fail = %d\n", 
		(end.tv_sec - start.tv_sec) + (double) (end.tv_nsec - start.tv_nsec) / 1000000000,
		sum, succ_1, succ_2, fail);
}
