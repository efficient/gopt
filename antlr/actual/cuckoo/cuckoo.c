#include "cuckoo.h"

int hash(int u)
{
	return CityHash32((char *) &u, 4);
}

void cuckoo_init(int **keys, struct cuckoo_bkt **ht_index)
{
	int i, slot_i;

	int bkt, key, success, failed_inserts = 0;

	/** < Allocate the hash table */
	printf("\tInitializing cuckoo index of size = %lu bytes\n", 
		NUM_BKT * sizeof(struct cuckoo_bkt));

	int sid = shmget(CUCKOO_KEY, NUM_BKT * sizeof(struct cuckoo_bkt), 
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("\tCould not create cuckoo hash index\n");
		exit(-1);
	}

	*ht_index = shmat(sid, 0, 0);
	memset((char *) *ht_index, 0, NUM_BKT * sizeof(struct cuckoo_bkt));

	/** < Allocate the packets and put them into the hash index randomly */
	printf("\tPutting %d keys into hash index randomly\n", NUM_KEYS);
	*keys = malloc(NUM_KEYS * sizeof(int));

	for(i = 0; i < NUM_KEYS; i++) {
		success = 0;

		// Generate a new key at random
		key = rand();
		assert(key != 0);
		(*keys)[i] = key;

		// Choose a random bucket
		bkt = hash(key) & NUM_BKT_;
		if(rand() % 2 == 0) {
			bkt = hash(bkt) & NUM_BKT_;
		}

		for(slot_i = 0; slot_i < 8; slot_i ++) {
			// Find an empty slot
			if((*ht_index)[bkt].slot[slot_i].key == 0) {
				(*ht_index)[bkt].slot[slot_i].key = key;
				(*ht_index)[bkt].slot[slot_i].value = i;
				success = 1;
				break;
			}
		}

		if(success == 0) {
			failed_inserts ++;
		}
	}

	printf("\tFraction of failed inserts = %f\n", 
		(double) failed_inserts / NUM_KEYS);
}


void red_printf(const char *format, ...)
{	
	#define RED_LIM 1000
	va_list args;
	int i;

	char buf1[RED_LIM], buf2[RED_LIM];
	memset(buf1, 0, RED_LIM);
	memset(buf2, 0, RED_LIM);

    va_start(args, format);

	// Marshal the stuff to print in a buffer
	vsnprintf(buf1, RED_LIM, format, args);

	// Probably a bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf1[i] == 0);
	}

	// Add markers for red color and reset color
	snprintf(buf2, 1000, "\033[31m%s\033[0m", buf1);

	// Probably another bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf2[i] == 0);
	}

	printf("%s", buf2);

    va_end(args);
}

