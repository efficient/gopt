#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<pthread.h>
#include<papi.h>
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

	/** < Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	red_printf("main: Initializing nodes for random walk\n");
	rand_walk_init(&nodes);

	red_printf("main: Starting random walks\n");
	/** < Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	/** < Do a random-walk from every node in the graph */
	for(i = 0; i < NUM_NODES; i += BATCH_SIZE) {
		process_batch(&nodes[i]);
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	red_printf("Time = %.4f, rate = %.2f sum = %lld\n"
		"Instructions = %lld, IPC = %f\n",
		real_time, NUM_NODES / real_time, sum,
		ins, ipc);

	return 0;
}
