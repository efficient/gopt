#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<pthread.h>
#include<time.h>

#include "fpp.h"
#include "rand-walk.h"

struct node *nodes;

long long sum = 0;

// batch_index must be declared outside process_batch
int batch_index = 0;

void process_batch(struct node *nodes) 
{
	int i, batch_index, next_nbh;
	struct node *cur_node[BATCH_SIZE];
		
	for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
		cur_node[batch_index] = &nodes[batch_index];
	}

	for(i = 0; i < STEPS; i ++) {
		for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
			sum += cur_node[batch_index]->id;

			/** < Compute the next neighbor */
			next_nbh = -1;
			while(next_nbh < 0) {
				next_nbh = rand() % 7;
			}
		
			cur_node[batch_index] = 
				(struct node *) nodes[batch_index].neighbors[next_nbh];
			__builtin_prefetch(cur_node[batch_index], 0, 0);
		}
	}
		
}

int main(int argc, char **argv)
{
	int i;
	struct timespec start, end;
	double seconds;

	red_printf("main: Initializing nodes for random walk\n");
	rand_walk_init(&nodes);

	red_printf("main: Starting random walks\n");
	clock_gettime(CLOCK_REALTIME, &start);

	/** < Do a random-walk from every node in the graph */
	for(i = 0; i < NUM_NODES; i += BATCH_SIZE) {
		process_batch(&nodes[i]);
	}

	clock_gettime(CLOCK_REALTIME, &end);

	seconds = (end.tv_sec - start.tv_sec) + 
		(double) (end.tv_nsec - start.tv_nsec) / 1000000000;
	red_printf("Time = %.4f, rate = %.2f sum = %lld\n",
		seconds, NUM_NODES / seconds, sum);

	return 0;
}
