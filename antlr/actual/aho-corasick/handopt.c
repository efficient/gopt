#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<unistd.h>
#include<time.h>

#include "aho.h"
#include "util.h"
#include "fpp.h"

#define USE_PAPI 0

#if USE_PAPI == 1
#include<papi.h>
#endif

#define PATTERN_FILE "../../../data_dump/snort/snort_longest_contents_bytes_sort"
#define NUM_PKTS (32 * 1024)
#define PKT_SIZE 1500

struct pkt {
	uint8_t content[PKT_SIZE];
};

/**< Generate NUM_PKTS packets for testing. Each test packet is constructed
  *  by concatenating patterns that were inserted into the AC engine. */
struct pkt *gen_packets(struct aho_pattern *patterns, int num_patterns)
{
	int i;
	struct pkt *test_pkts = malloc(NUM_PKTS * sizeof(struct pkt));
	assert(test_pkts != NULL);
	memset(test_pkts, 0, NUM_PKTS * sizeof(struct pkt));

	for(i = 0; i < NUM_PKTS; i ++) {
		int index = 0;
		while(index < PKT_SIZE) {
			test_pkts[i].content[index] = rand() % AHO_ALPHA_SIZE;
			index ++;
		}
		/** Code for generating workload with concatenated content strings
		int tries = 0;
		while(tries < 10) {
			int pattern_i = rand() % num_patterns;

			if(index + patterns[pattern_i].len <= PKT_SIZE) {
				memcpy((char *) &(test_pkts[i].content[index]), 
					patterns[pattern_i].content, patterns[pattern_i].len);
				index += patterns[pattern_i].len;
				break;
			} else {
				tries ++;
			}
		} */
	}

	return test_pkts;
}

int batch_index = 0;
int tot_match = 0;

void process_batch(const struct aho_state *dfa,
	const uint8_t *terminal_states, const struct pkt *test_pkts)
{
	int j = 0, state[BATCH_SIZE] = {0};

	for(j = 0; j < PKT_SIZE; j ++) {
		for(batch_index = 0; batch_index < BATCH_SIZE; batch_index ++) {
			int inp = test_pkts[batch_index].content[j];

			if(terminal_states[state[batch_index]] == 1) {
				struct ds_qnode *t = dfa[state[batch_index]].output.head;
				while(t != NULL) {
					tot_match += t->data;
					t = t->next;
				}
			}

			state[batch_index] = dfa[state[batch_index]].G[inp];
		}
	}
}


int main(int argc, char *argv[])
{
	printf("%lu\n", sizeof(struct aho_state));
	int num_patterns, i, j;
	int *count;

	struct aho_state *dfa;
	uint8_t terminal_states[AHO_MAX_STATES] = {0};
	aho_init(&dfa);

	/**< Get the patterns */
	struct aho_pattern *patterns = aho_get_patterns(PATTERN_FILE, 
		&num_patterns);

	/**< Build the DFA */
	red_printf("Building AC goto function: \n");
	for(i = 0; i < num_patterns; i ++) {
		aho_add_pattern(dfa, &patterns[i], i);
	}

	red_printf("Building AC failure function\n");
	aho_build_ff(dfa);
	aho_preprocess_dfa(dfa, terminal_states);

	/**< Generate the workload packets */
	red_printf("Generating packets\n");
	struct pkt *test_pkts = gen_packets(patterns, num_patterns);

	red_printf("Starting lookups\n");
	assert(NUM_PKTS % BATCH_SIZE == 0);

#if USE_PAPI == 1

	/** < Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	/** < Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	for(i = 0; i < NUM_PKTS; i += BATCH_SIZE) {
		process_batch(dfa, &test_pkts[i]);
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	red_printf("Time = %.4f s, Instructions = %lld, IPC = %f, tot_match = %d\n",
		real_time, ins, ipc, tot_match);
	double ns = real_time * 1000000000;
	red_printf("Rate = %.2f Gbps.\n", ((double) NUM_PKTS * PKT_SIZE * 8) / ns);

#else

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for(i = 0; i < NUM_PKTS; i += BATCH_SIZE) {
		process_batch(dfa, terminal_states, &test_pkts[i]);
	}
	
	clock_gettime(CLOCK_REALTIME, &end);

	double ns = (end.tv_sec - start.tv_sec) * 1000000000 +
		(double) (end.tv_nsec - start.tv_nsec);
	red_printf("Rate = %.2f Gbps. tot_match = %d\n", 
		((double) NUM_PKTS * PKT_SIZE * 8) / ns, tot_match);

#endif
	

	/**< Clean up */
	for(i = 0; i < num_patterns; i ++) {
		free(patterns[i].content);
	}

	free(patterns);
	free(test_pkts);
	free(dfa);

	return 0;
}
