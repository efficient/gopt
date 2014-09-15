#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <assert.h>
#include "city.h"

/** < Number of cuckoo buckets */
#define NUM_BKT (8 * 1024 * 1024)
#define NUM_BKT_ ((8 * 1024 * 1024) - 1)

/** < Number of keys inserted into the hash table */
#define NUM_KEYS (16 * 1024 * 1024)
#define NUM_KEYS_ ((16 * 1024 * 1024) - 1)

/** < Key for shmget */
#define CUCKOO_KEY 1

struct cuckoo_slot
{
	int key;
	int value;
};

struct cuckoo_bkt
{
	struct cuckoo_slot slot[8];
};

int hash(int u);
void cuckoo_init(int **keys, struct cuckoo_bkt** ht_index);
void red_printf(const char *format, ...);

