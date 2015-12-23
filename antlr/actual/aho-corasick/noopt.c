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

#define DEBUG 0

/* Maximum number of patterns a packet can match during its DFA traversal */
#define MAX_MATCH 8192

/* A list of patterns matched by a packet */
struct mp_list_t {
	int num_match;
	uint16_t ptrn_id[MAX_MATCH];
};

/* Plain old API-call batching */
void process_batch(const struct aho_dfa *dfa_arr,
	const struct aho_pkt *pkts, struct mp_list_t *mp_list)
{
	int I, j;

	for(I = 0; I < BATCH_SIZE; I++) {
		int dfa_id = pkts[I].dfa_id;
		int len = pkts[I].len;
		struct aho_state *st_arr = dfa_arr[dfa_id].root;
		
		int state = 0;

		for(j = 0; j < len; j++) {
			int count = st_arr[state].output.count;

			if(count != 0) {
				/* This state matches some patterns: copy the pattern IDs
				  *  to the output */
				int offset = mp_list[I].num_match;
				memcpy(&mp_list[I].ptrn_id[offset],
					st_arr[state].out_arr, count * sizeof(uint16_t));

				mp_list[I].num_match += count;
			}
			int inp = pkts[I].content[j];
			state = st_arr[state].G[inp];
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

	/* Per-batch matched patterns */
	struct mp_list_t mp_list[BATCH_SIZE];
	for(i = 0; i < BATCH_SIZE; i++) {
		mp_list[i].num_match = 0;
	}

	/* Being paranoid about GCC optimization: ensure that the memcpys in
	  *  process_batch functions don't get optimized out */
	int matched_pat_sum = 0;

	int tot_proc = 0;		/* How many packets did we actually match ? */
	int tot_success = 0;	/* Packets that matched a DFA state */ 
	int tot_bytes = 0;		/* Total bytes matched through DFAs */

	while(1) {
		struct timespec start, end;
		clock_gettime(CLOCK_REALTIME, &start);

		for(i = 0; i < num_pkts; i += BATCH_SIZE) {
			process_batch(dfa_arr, &pkts[i], mp_list);

			for(j = 0; j < BATCH_SIZE; j++) {
				int num_match = mp_list[j].num_match;
				assert(num_match < MAX_MATCH);

				tot_success += num_match == 0 ? 0 : 1;

				int pat_i;

				#if DEBUG == 1
				printf("Pkt %d matched: ", pkts[i + j].pkt_id);

				for(pat_i = 0; pat_i < num_match; pat_i++) {
					printf("%d ", mp_list[j].ptrn_id[pat_i]);
					matched_pat_sum += mp_list[j].ptrn_id[pat_i];
				}

				printf("\n");
				#else
				for(pat_i = 0; pat_i < num_match; pat_i++) {
					matched_pat_sum += mp_list[j].ptrn_id[pat_i];
				}
				#endif

				tot_proc++;
				tot_bytes += pkts[i + j].len;

				/* Re-initialize for next iteration */
				mp_list[j].num_match = 0;
			}
		}

		clock_gettime(CLOCK_REALTIME, &end);

		double ns = (end.tv_sec - start.tv_sec) * 1000000000 +
			(double) (end.tv_nsec - start.tv_nsec);
		red_printf("ID %d: Rate = %.2f Gbps. tot_success = %d\n", id,
			((double) tot_bytes * 8) / ns, tot_success);
		red_printf("num_pkts = %d, tot_proc = %d | matched_pat_sum = %d\n",
			num_pkts, tot_proc, matched_pat_sum);

		matched_pat_sum = 0;	/* Sum of all matched pattern IDs */
		tot_success = 0;
		tot_bytes = 0;
		tot_proc = 0;

		#if DEBUG == 1		/* Print matched states only once */
		exit(0);
		#endif
	}
}

int main(int argc, char *argv[])
{
	assert(argc == 2);

	int num_threads = atoi(argv[1]);
	assert(num_threads >= 1 && num_threads <= AHO_MAX_THREADS);

	int num_patterns, num_pkts, i;

	struct aho_pattern *patterns;
	struct aho_pkt *pkts;
	struct aho_dfa dfa_arr[AHO_MAX_DFA];

	/* Thread structures */
	pthread_t worker_threads[AHO_MAX_THREADS];
	struct aho_ctrl_blk worker_cb[AHO_MAX_THREADS];

	red_printf("State size = %lu\n", sizeof(struct aho_state));

	/* Initialize the shared DFAs */
	for(i = 0; i < AHO_MAX_DFA; i++) {
		printf("Initializing DFA %d\n", i);
		aho_init(&dfa_arr[i], i);
	}

	red_printf("Adding patterns to DFAs\n");
	patterns = aho_get_patterns(AHO_PATTERN_FILE,
		&num_patterns);

	for(i = 0; i < num_patterns; i++) {
		int dfa_id = patterns[i].dfa_id;
		aho_add_pattern(&dfa_arr[dfa_id], &patterns[i], i);
	}

	red_printf("Building AC failure function\n");
	for(i = 0; i < AHO_MAX_DFA; i++) {
		aho_build_ff(&dfa_arr[i]);
		aho_preprocess_dfa(&dfa_arr[i]);
	}

	red_printf("Reading packets from file\n");
	pkts = aho_get_pkts(AHO_PACKET_FILE, &num_pkts);
	
	for(i = 0; i < num_threads; i++) {
		worker_cb[i].tid = i;
		worker_cb[i].dfa_arr = dfa_arr;
		worker_cb[i].pkts = pkts;
		worker_cb[i].num_pkts = num_pkts;

		pthread_create(&worker_threads[i], NULL, ids_func, &worker_cb[i]);

		/* Ensure that threads don't use the same packets close in time */
		sleep(1);
	}

	for(i = 0; i < num_threads; i++) {
		pthread_join(worker_threads[i], NULL);
	}

	/* The work never ends */
	assert(0);

	return 0;
}
