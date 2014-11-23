#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<unistd.h>
#include<time.h>
#include<pthread.h>

#include "aho.h"
#include "util.h"
#include "fpp.h"

#define NUM_PKTS (64 * 1024)
#define PKT_SIZE 1518

struct pkt {
	uint8_t content[PKT_SIZE];
};

/**< Generate NUM_PKTS packets for lookups */
struct pkt *gen_packets(int seed)
{
	int i;
	srand(seed);

	struct pkt *pkts = malloc(NUM_PKTS * sizeof(struct pkt));
	assert(pkts != NULL);
	memset(pkts, 0, NUM_PKTS * sizeof(struct pkt));

	for(i = 0; i < NUM_PKTS; i ++) {
		int index = 0;
		while(index < PKT_SIZE) {
			pkts[i].content[index] = rand() % AHO_ALPHA_SIZE;
			index ++;
		}
	}

	return pkts;
}

int process_single(struct aho_state *state_arr, struct pkt *test_pkt)
{
	int j = 0, state = 0;

	for(j = 0; j < PKT_SIZE; j ++) {
		if(state_arr[state].output.count != 0) {
			return 1;
		}

		int inp = test_pkt->content[j];
		state = state_arr->G[inp];
	}

	return 0;
}

void process_batch(struct aho_dfa *dfa,
	struct pkt *test_pkts, int *success)
{
	int batch_index = 0;
	int j = 0, state[BATCH_SIZE] = {0};
	struct aho_state *st_arr = dfa->root;

	for(j = 0; j < PKT_SIZE; j ++) {
		for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
			if(st_arr[state[batch_index]].output.count != 0) {
				success[batch_index] ++;
			}

			int inp = test_pkts[batch_index].content[j];
			state[batch_index] = st_arr[state[batch_index]].G[inp];
		}
	}
}


void *ids_func(void *ptr)
{
	int i, j;
	struct aho_ctrl_blk *cb = (struct aho_ctrl_blk *) ptr;
	int id = cb->tid;
	red_printf("Starting thread %d. Generating packets..\n", id);

	/**< Aho-Corasick specific structures */
	struct pkt *test_pkts = gen_packets(id);
	red_printf("Thread %d. Done generating packets..\n", id);
	
	int success[BATCH_SIZE] = {0};
	int tot_success = 0;

	/**< Shared structures */
	struct aho_dfa *dfa_arr = cb->dfa_arr;

	while(1) {
		struct timespec start, end;
		clock_gettime(CLOCK_REALTIME, &start);

		for(i = 0; i < NUM_PKTS; i += BATCH_SIZE) {
			memset(success, 0, BATCH_SIZE * sizeof(int));

			int dfa_index = rand() % AHO_MAX_DFA;

			process_batch(&dfa_arr[dfa_index], &test_pkts[i], success);

			for(j = 0; j < BATCH_SIZE; j ++) {
				tot_success += (success[j] == 0 ? 0 : 1);
			}
		}

		clock_gettime(CLOCK_REALTIME, &end);

		double ns = (end.tv_sec - start.tv_sec) * 1000000000 +
			(double) (end.tv_nsec - start.tv_nsec);
		red_printf("ID %d: Rate = %.2f Gbps. tot_success = %d\n", id,
			((double) NUM_PKTS * PKT_SIZE * 8) / ns, tot_success);

		tot_success = 0;
	}
}

int main(int argc, char *argv[])
{
	printf("%lu\n", sizeof(struct aho_state));

	/**< Sanity checks */
	assert(argc == 2);
	assert(NUM_PKTS % BATCH_SIZE == 0);

	int num_threads = atoi(argv[1]);
	assert(num_threads >= 1 && num_threads <= AHO_MAX_THREADS);
	
	int num_patterns, i;

	/**< Shared Aho-Corasick structures */
	struct aho_dfa dfa_arr[AHO_MAX_DFA];

	/**< Thread structures */
	pthread_t worker_threads[AHO_MAX_THREADS];
	struct aho_ctrl_blk worker_cb[AHO_MAX_THREADS];

	/**< Get the patterns */
	struct aho_pattern *patterns = aho_get_patterns(AHO_PATTERN_FILE,
		&num_patterns);

	/**< Initialize the shared DFAs */
	for(i = 0; i < AHO_MAX_DFA; i ++) {
		printf("Initializing DFA %d\n", i);
		aho_init(&dfa_arr[i], i);
	}

	/**< Insert patterns into the DFAs */
	red_printf("Adding patterns to DFAs\n");
	for(i = 0; i < num_patterns; i ++) {
		int dfa_id = patterns[i].dfa_id;
		aho_add_pattern(&dfa_arr[dfa_id], &patterns[i], i);
	}

	red_printf("\tBuilding AC failure function\n");
	for(i = 0; i < AHO_MAX_DFA; i ++) {
		aho_build_ff(&dfa_arr[i]);
		aho_preprocess_dfa(&dfa_arr[i]);
	}

	for(i = 0; i < num_threads; i ++) {
		worker_cb[i].tid = i;
		worker_cb[i].dfa_arr = dfa_arr;
		pthread_create(&worker_threads[i], NULL, ids_func, &worker_cb[i]);

		/**< Avoid libc lock contention during rand() calls in gen_packets */
		sleep(4);
	}

	for(i = 0; i < num_threads; i ++) {
		pthread_join(worker_threads[i], NULL);
	}

	/**< The work never ends */
	assert(0);

	return 0;
}
