#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <assert.h>

/** < 512 MB: RAM */
#define NUM_NODES (8 * 1024 * 1024)
#define NUM_NODES_ (NUM_NODES - 1)

/** < Key for shmget */
#define RAND_WALK_KEY 1

/** < Number of random-walk steps */
#define STEPS 10

struct node
{
	long long id;
	void *neighbors[7];
};

void rand_walk_init(struct node **nodes);
void red_printf(const char *format, ...);

