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

#define DFA_BATCH_SIZE 1024

struct dfa_batch_t {
	int batch_size;			/**< Number of items in this batch */
	int tot_bytes;			/**< Total length of packets in this batch */
	int success[DFA_BATCH_SIZE];		/**< After sorting */
	struct aho_pkt pkts[DFA_BATCH_SIZE];
};

int compare(const void *p1, const void *p2)
{
	const struct aho_pkt *pkt_1 = p1;
	const struct aho_pkt *pkt_2 = p2;

	if(pkt_1->len < pkt_2->len) {
		return -1;
	} else if(pkt_1->len > pkt_2->len) {
		return 1;
	}

	return 0;
}

/*int process_single(struct aho_state *state_arr, struct pkt *test_pkt)
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
}*/

void process_batch(const struct aho_dfa *dfa,
	const struct aho_pkt *pkts, int *success)
{
	int j = 0, I = 0, state[BATCH_SIZE] = {0};
	struct aho_state *st_arr = dfa->root;

	/**< Find the longest packet in this batch */

	/*
	int max_len = 0, avg_len = 0, min_len = 10000000;
	for(I = 0; I < BATCH_SIZE; I ++) {
		max_len = pkts[I].len > max_len ? pkts[I].len : max_len;
		min_len = pkts[I].len < min_len ? pkts[I].len : min_len;
		avg_len += pkts[I].len;
	}

	avg_len /= BATCH_SIZE;
	printf("\tmax = %d, min = %d, avg = %d\n", max_len, min_len, avg_len);
	*/

	int max_len = 0;
	for(I = 0; I < BATCH_SIZE; I ++) {
		//printf("%d ", pkts[I].len);
		max_len = pkts[I].len > max_len ? pkts[I].len : max_len;
	}

	//printf("\n");
	//usleep(10000);

	for(j = 0; j < max_len; j ++) {
		for(I = 0; I < BATCH_SIZE; I ++) {
			if(st_arr[state[I]].output.count != 0) {
				success[I] ++;
			}

			if(j >= pkts[I].len) {
				continue;
			}

			int inp = pkts[I].content[j];
			state[I] = st_arr[state[I]].G[inp];
		}
	}
}

void *ids_func(void *ptr)
{
	int i, j;

	struct aho_ctrl_blk *cb = (struct aho_ctrl_blk *) ptr;
	int id = cb->tid;
	struct aho_dfa *dfa_arr = cb->dfa_arr;
	struct aho_pkt *pkts = cb->pkts;
	int num_pkts = cb->num_pkts;

	red_printf("Starting thread %d", id);

	struct dfa_batch_t *dfa_batch = malloc(AHO_MAX_DFA * sizeof(struct dfa_batch_t));
	memset(dfa_batch, 0, AHO_MAX_DFA * sizeof(struct dfa_batch_t));

	int tot_proc = 0, tot_success = 0, tot_bytes;

	while(1) {
		struct timespec start, end;
		clock_gettime(CLOCK_REALTIME, &start);

		for(i = 0; i < num_pkts; i ++) {
			int dfa_id = pkts[i].dfa_id;

			if(dfa_batch[dfa_id].batch_size == DFA_BATCH_SIZE) {

				qsort(dfa_batch[dfa_id].pkts,
					DFA_BATCH_SIZE, sizeof(struct aho_pkt), compare);

				for(j = 0; j < DFA_BATCH_SIZE; j += BATCH_SIZE) {
					process_batch(&dfa_arr[dfa_id], 
						&dfa_batch[dfa_id].pkts[j], &dfa_batch[dfa_id].success[j]);
				}

				tot_proc += DFA_BATCH_SIZE;

				for(j = 0; j < DFA_BATCH_SIZE; j ++) {
					tot_success += (dfa_batch[dfa_id].success[j] == 0 ? 0 : 1);
				}

				memset(dfa_batch[dfa_id].success, 0, DFA_BATCH_SIZE * sizeof(int));

				tot_bytes += dfa_batch[dfa_id].tot_bytes;

				dfa_batch[dfa_id].batch_size = 0;
				dfa_batch[dfa_id].tot_bytes = 0;

			} else {
				int batch_index = dfa_batch[dfa_id].batch_size;
				dfa_batch[dfa_id].pkts[batch_index] = pkts[i];		/**< Shallow copy */
				dfa_batch[dfa_id].tot_bytes += pkts[i].len;
			}

			/**< Add the new packet to this DFA batch */
			dfa_batch[dfa_id].batch_size ++;
		}

		clock_gettime(CLOCK_REALTIME, &end);

		double ns = (end.tv_sec - start.tv_sec) * 1000000000 +
			(double) (end.tv_nsec - start.tv_nsec);
		red_printf("ID %d: Rate = %.2f Gbps. tot_success = %d\n", id,
			((double) tot_bytes * 8) / ns, tot_success);
		red_printf("num_pkts = %d, tot_proc = %d\n", num_pkts, tot_proc);

		tot_success = 0;
		tot_bytes = 0;
		tot_proc = 0;
	}
}

int main(int argc, char *argv[])
{
	assert(argc == 2);
	assert(DFA_BATCH_SIZE % BATCH_SIZE == 0);

	int num_threads = atoi(argv[1]);
	assert(num_threads >= 1 && num_threads <= AHO_MAX_THREADS);

	int num_patterns, num_pkts, i;

	struct aho_pattern *patterns;
	struct aho_pkt *pkts;
	struct aho_dfa dfa_arr[AHO_MAX_DFA];

	/**< Thread structures */
	pthread_t worker_threads[AHO_MAX_THREADS];
	struct aho_ctrl_blk worker_cb[AHO_MAX_THREADS];

	red_printf("State size = %lu\n", sizeof(struct aho_state));


	/**< Initialize the shared DFAs */
	for(i = 0; i < AHO_MAX_DFA; i ++) {
		printf("Initializing DFA %d\n", i);
		aho_init(&dfa_arr[i], i);
	}

	red_printf("Adding patterns to DFAs\n");
	patterns = aho_get_patterns(AHO_PATTERN_FILE,
		&num_patterns);

	for(i = 0; i < num_patterns; i ++) {
		int dfa_id = patterns[i].dfa_id;
		aho_add_pattern(&dfa_arr[dfa_id], &patterns[i], i);
	}

	red_printf("Building AC failure function\n");
	for(i = 0; i < AHO_MAX_DFA; i ++) {
		aho_build_ff(&dfa_arr[i]);
		aho_preprocess_dfa(&dfa_arr[i]);
	}

	red_printf("Reading packets from file\n");
	pkts = aho_get_pkts(AHO_PACKET_FILE, &num_pkts);
	
	for(i = 0; i < num_threads; i ++) {
		worker_cb[i].tid = i;
		worker_cb[i].dfa_arr = dfa_arr;
		worker_cb[i].pkts = pkts;
		worker_cb[i].num_pkts = num_pkts;

		pthread_create(&worker_threads[i], NULL, ids_func, &worker_cb[i]);

		/**< Ensure that threads don't use the same packets close in time */
		sleep(4);
	}

	for(i = 0; i < num_threads; i ++) {
		pthread_join(worker_threads[i], NULL);
	}

	/**< The work never ends */
	assert(0);

	return 0;
}
