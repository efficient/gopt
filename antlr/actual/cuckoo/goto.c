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

// City hash of an unsigned number
uint32_t cityhash(uint32_t u)
{
	return CityHash32((char *) &u, 4);
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

#include "fpp.h"
void process_pkts_in_batch(uint32_t *pkt_lo)
{
	uint32_t K[BATCH_SIZE];
	uint32_t S1[BATCH_SIZE];
	uint32_t S2[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&label_0;
	}

label_0:

        // Try the first slot
        K[I] = pkt_lo[I];
        S1[I] = cityhash(K[I]) % HASH_INDEX_N;
        FPP_PSS(&hash_index[S1[I]], label_1);
label_1:

        if(hash_index[S1[I]].key == K[I]) {
            sum += hash_index[S1[I]].value;
            succ_1 ++;
        } else {
            // Try the second slot
            S2[I] = cityhash(K[I] + 1) % HASH_INDEX_N;
            FPP_PSS(&hash_index[S2[I]], label_2);
label_2:

            if(hash_index[S2[I]].key == K[I]) {
                sum += hash_index[S2[I]].value;
                succ_2 ++;
            } else {
                fail ++;
            }
        }
       
end:
    batch_rips[I] = &&end;
    iMask = FPP_SET(iMask, I); 
    if(iMask == (1 << BATCH_SIZE) - 1) {
        return;
    }
    I = (I + 1) & BATCH_SIZE_;
    goto *batch_rips[I];

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
