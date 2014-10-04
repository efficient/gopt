#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<pthread.h>
#include<stdint.h>

#include "city.h"
#include "util.h"

#define NUM_THREADS 4
#define WRITER_COMPUTE 1

#define NUM_LOCKS 1024
#define NUM_LOCKS_ (NUM_LOCKS - 1)

#define NUM_NODES (1024 * 1024)
#define NUM_NODES_ (NUM_NODES - 1)

#define GHZ_CPS 1000000000
#define ITERS_PER_MEASUREMENT 10000000

typedef struct {
	long long a;
	long long b;
} node_t;

typedef struct {
	volatile long long lock;
	long long pad[7];
} lock_t;

void *reader(void *ptr);
void *writer(void *ptr);

/** < Only shared variables here */
node_t *nodes;
lock_t *locks;

int main()
{
	int i;
	int tid[NUM_THREADS];
	pthread_t thread[NUM_THREADS];

	/** < Ensure that locks are cacheline aligned */
	assert(sizeof(lock_t) == 64);

	/** < Allocate the shared nodes */
	red_printf("Allocting %d nodes\n", NUM_NODES);
	nodes = (node_t *) malloc(NUM_NODES * sizeof(node_t));
	assert(nodes != NULL);
	
	for(i = 0; i < NUM_NODES; i ++) {
		nodes[i].a = rand();
		nodes[i].b = nodes[i].a + 1;
	}

	/** < Allocate the striped spinlocks */
	red_printf("Allocting %d locks\n", NUM_LOCKS);
	locks = (lock_t *) malloc(NUM_LOCKS * sizeof(lock_t));
	assert(locks != NULL);
	
	for(i = 0; i < NUM_LOCKS; i++) {
		locks[i].lock = 0;
	}
	
	/** < Launch several reader threads and a writer thread */
	for(i = 0; i < NUM_THREADS; i++) {
		tid[i] = i;
		if(i == 0) {
			red_printf("Launching writer thread with tid = %d\n", tid[i]);
			pthread_create(&thread[i], NULL, writer, &tid[i]);
		} else {
			red_printf("Launching reader thread with tid = %d\n", tid[i]);
			pthread_create(&thread[i], NULL, reader, &tid[i]);
		}
	}

	for(i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread[i], NULL);
	}

	exit(0);
}

void *reader( void *ptr)
{
	struct timespec start, end;
	int tid = *((int *) ptr);
	uint64_t seed = 0xdeadbeef + tid;
	int sum = 0;

	/** < The node and lock to use in an iteration */
	int node_id, lock_id;

	/** < The snapshotted lock version and node data */
	int lock_version;
	node_t node_snapshot;

	/** < Total number of times we start the snapshotting procedure */
	int num_tries = 0;

	/** < Total number of iterations (for measurement) */
	int num_iters = 0;

	clock_gettime(CLOCK_REALTIME, &start);

	while(1) {
		if(num_iters == ITERS_PER_MEASUREMENT) {
			clock_gettime(CLOCK_REALTIME, &end);
			double seconds = (end.tv_sec - start.tv_sec) + 
				(double) (end.tv_nsec - start.tv_nsec) / GHZ_CPS;
		
			printf("Reader %d: rate = %.2f M/s. Sum = %d. Avg. tries = %f\n", tid, 
				num_iters / (1000000 * seconds), sum, (double) num_tries / num_iters);
				
			num_iters = 0;
			num_tries = 0;

			clock_gettime(CLOCK_REALTIME, &start);
		}

		node_id = fastrand(&seed) & NUM_NODES_;
		lock_id = node_id & NUM_LOCKS_;

try_again:
		num_tries ++;

		/** < Enter the critical section when the version is even */
		lock_version = locks[lock_id].lock; 
		if((lock_version & 1) != 0) {
			goto try_again;
		}
	
		/** < version load #1 --> snapshot loads */
		asm volatile("" ::: "memory");

		node_snapshot.a = nodes[node_id].a;
		node_snapshot.b = nodes[node_id].b;
	
		/** < snapshot loads --> version load #2 */
		asm volatile("" ::: "memory");

		if(locks[lock_id].lock == lock_version) {
			// Snapshot was correct
			assert(node_snapshot.b == node_snapshot.a + 1);
			sum += node_snapshot.a + node_snapshot.b;
		} else {
			goto try_again;
		}

		num_iters ++;
	}
}


void *writer( void *ptr)
{
	struct timespec start, end;
	int tid = *((int *) ptr);
	uint64_t seed = 0xdeadbeef + tid;
	int sum = 0;

	/** < The node and lock to use in an iteration */
	int i, node_id, lock_id;
	
	/** < Total number of iterations (for measurement) */
	int num_iters = 0;

	clock_gettime(CLOCK_REALTIME, &start);

	while(1) {
		if(num_iters == ITERS_PER_MEASUREMENT) {
			clock_gettime(CLOCK_REALTIME, &end);
			double seconds = (end.tv_sec - start.tv_sec) + 
				(double) (end.tv_nsec - start.tv_nsec) / GHZ_CPS;
		
			node_id = fastrand(&seed) & NUM_NODES_;

			red_printf("Writer thread %d: rate = %.2f M/s. "
				"Random node: (%lld, %lld)\n", tid, 
				num_iters / (1000000 * seconds),
				nodes[node_id].a, nodes[node_id].b);
				
			num_iters = 0;
			clock_gettime(CLOCK_REALTIME, &start);
		}

		node_id = fastrand(&seed) & NUM_NODES_;
		lock_id = node_id & NUM_LOCKS_;

		locks[lock_id].lock ++;

		/** < version store #1 --> node stores */
		asm volatile("" ::: "memory");

		/** < Update node.a and node.b after some expensive computation */
		for(i = 0; i < WRITER_COMPUTE; i ++) {
			nodes[node_id].a = CityHash32((char *) &nodes[node_id].a, 4);
		}

		nodes[node_id].b = nodes[node_id].a + 1;
		
		/** < node stores --> version store #2 */
		asm volatile("" ::: "memory");

		locks[lock_id].lock ++;

		num_iters ++;
	}
}
