#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<unistd.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "aho.h"
#include "util.h"

/*
 * DARPA dataset has a lot of packets with identical payload. Enable this
 * to use random bytes in packets.
 */
#define USE_RANDOM_PAYLOAD 1

/* Initialize the state transition table and o/p queues */
void aho_init(struct aho_dfa *dfa, int id)
{
	assert(dfa != NULL);

	dfa->id = id;
	dfa->num_used_states = 0;

	int sid = shmget(AHO_SHM_KEY + id, 
		AHO_MAX_STATES * sizeof(struct aho_state),
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("\tCould not allocate states for DFA %d\n", id);
		exit(-1);
	}

	dfa->root = shmat(sid, 0, 0);
	assert(dfa->root != NULL);

	int i;
	for(i = 0; i < AHO_MAX_STATES; i++) {
		struct aho_state *cur_state = &dfa->root[i];
		memset(cur_state->G, AHO_FAIL, AHO_ALPHA_SIZE * sizeof(uint16_t));
		cur_state->F = AHO_FAIL;
		ds_queue_init(&cur_state->output);
	}
}

/* Add a pattern to the DFA */
void
aho_add_pattern(struct aho_dfa *dfa, struct aho_pattern *pattern, int index)
{
	int length = pattern->len;
	assert(length >= 0 && length <= 100000);

	struct aho_state *st_arr = dfa->root;
	int j, state = 0;

	for(j = 0; j < length; j++) {
		if(st_arr[state].G[pattern->content[j]] == AHO_FAIL) {
			break;
		}
		state = st_arr[state].G[pattern->content[j]];
	}

	/* Characters j to (length - 1) need new states */
	for(; j < length; j++) {
		dfa->num_used_states++;

		/* AHO_MAX_STATES - 1 is AHO_FAIL */
		assert(dfa->num_used_states <= AHO_MAX_STATES - 2);

		/* Print when states consume around 30 MB */
		if(dfa->num_used_states % 10000 == 0) {
			printf("\taho: DFA %d states = %d\n", dfa->id, dfa->num_used_states);
		}

		st_arr[state].G[pattern->content[j]] = dfa->num_used_states;
		state = dfa->num_used_states;
	}

	/* Add this pattern as the output for the last state */
	assert(state != 0);
	ds_queue_add(&st_arr[state].output, index);
}

/* Build the Aho-Coraick failure function */
void aho_build_ff(struct aho_dfa *dfa)
{
	int i;
	struct ds_queue state_queue;
	ds_queue_init(&state_queue);

	struct aho_state *st_arr = dfa->root;

	/* Invalid transitions from the root state need to loop back */
	for(i = 0; i < AHO_ALPHA_SIZE; i++) {
		if(st_arr[0].G[i] == AHO_FAIL) {
			st_arr[0].G[i] = 0;
		}
	}

	/* Initialize the failure function of the root's children */
	for(i = 0; i < AHO_ALPHA_SIZE; i++) {
		int next_state = st_arr[0].G[i];
		if(next_state != 0) {
			ds_queue_add(&state_queue, next_state);
			st_arr[next_state].F = 0;
		}
	}

	/* Create the failure function recursively */
	while(!ds_queue_is_empty(&state_queue)) {
		int cur_state = ds_queue_remove(&state_queue);

		/* Look at all the valid state transitions from cur_state */
		for(i = 0; i < AHO_ALPHA_SIZE; i++) {
			int child_state = st_arr[cur_state].G[i];

			if(child_state != AHO_FAIL) {
				ds_queue_add(&state_queue, child_state);

				int best_state = st_arr[cur_state].F;
				while(st_arr[best_state].G[i] == AHO_FAIL) {
					best_state = st_arr[best_state].F;
				}

				int child_fail_state = st_arr[best_state].G[i];
				st_arr[child_state].F = child_fail_state;

				/* Add all patterns from child_state.F to child_state */
				struct ds_qnode *t = st_arr[child_fail_state].output.head;
				while(t != NULL) {
					ds_queue_add(&st_arr[child_state].output, t->data);
					t = t->next;
				}
			}
		}	/* End loop over children states */
	}	/* Finish traversal */
}

/* Get variable length strings from a pattern file */
struct aho_pattern 
*aho_get_strings(const char *pattern_file, int *num_patterns)
{
	assert(pattern_file != NULL && num_patterns != NULL);

	int i;
	struct aho_pattern *patterns;
	size_t buf_size;
	char *first_newline = NULL;

	FILE *pattern_fp = fopen(pattern_file, "r");
	assert(pattern_fp != NULL);

	/* Get the number of patterns and do a sanity check */
	fscanf(pattern_fp, "%d", num_patterns);
	assert(*num_patterns >= 0 && *num_patterns <= AHO_MAX_PATTERNS);
	printf("\taho: num_patterns = %d\n", *num_patterns);

	/* Get the newline after num_patterns (otherwise getline() reads it) */
	getline(&first_newline, &buf_size, pattern_fp);

	/* Initialize pattern pointers: input to getline() should ne NULL */
	patterns = (struct aho_pattern *) malloc(*num_patterns * 
		sizeof(struct aho_pattern));
	assert(patterns != NULL);
	memset(patterns, 0, *num_patterns * sizeof(struct aho_pattern));

	/* Read the actual patterns from */
	for(i = 0; i < *num_patterns; i++) {
		int num_chars = getline((char **) &patterns[i].content, 
			(size_t *) &buf_size, pattern_fp);

		patterns[i].len = num_chars - 1;
		assert(patterns[i].len <= AHO_MAX_PATTERN_LEN);

		/* Zero out the newline at the end */
		patterns[i].content[num_chars - 1] = 0;
	}

	return patterns;
}

/*
 * Get Snort's content strings from a byte-file. File format:
 * <num_patterns>
 * <dfa id> <num_bytes> byte_1 byte_2 ...
 * ...
 */
struct aho_pattern 
*aho_get_patterns(const char *pattern_file, int *num_patterns)
{
	assert(pattern_file != NULL && num_patterns != NULL);

	int i, j;
	struct aho_pattern *patterns;

	int dfa_load[AHO_MAX_DFA] = {0};

	FILE *pattern_fp = fopen(pattern_file, "r");
	assert(pattern_fp != NULL);

	/* Get the number of patterns and do a sanity check */
	fscanf(pattern_fp, "%d", num_patterns);
	assert(*num_patterns >= 0 && *num_patterns <= AHO_MAX_PATTERNS);
	printf("\taho: num_patterns = %d\n", *num_patterns);

	/* Initialize pattern pointers */
	patterns = (struct aho_pattern *) malloc(*num_patterns * 
		sizeof(struct aho_pattern));
	assert(patterns != NULL);
	memset(patterns, 0, *num_patterns * sizeof(struct aho_pattern));

	/* Get the actual content strings */
	for(i = 0; i < *num_patterns; i++) {
		int dfa_id;
		int len;

		/* Get the DFA ID of this pattern */
		fscanf(pattern_fp, "%d", &dfa_id);
		assert(dfa_id >= 0 && dfa_id < AHO_MAX_DFA);
		patterns[i].dfa_id = dfa_id;

		/* Get the length of this pattern */
		fscanf(pattern_fp, "%d", &len);
		assert(len >= 0 && len < AHO_MAX_PATTERN_LEN);
		patterns[i].len = len;		

		patterns[i].content = malloc(len);
		assert(patterns[i].content != NULL);

		/* Get one byte at a time */
		for(j = 0; j < len; j++) {
			int cur_byte;
			fscanf(pattern_fp, "%d", &cur_byte);
			assert(cur_byte >= 0 && cur_byte <= 255);
			patterns[i].content[j] = (uint8_t) cur_byte;
		}

		dfa_load[dfa_id]++;
	}

	printf("\taho: Printing DFAs with > 130 patterns\n");
	for(i = 0; i < AHO_MAX_DFA; i++) {
		if(dfa_load[i] > 130) {
			printf("\t\taho: DFA %d has %d patterns\n", i, dfa_load[i]);
		}
	}

	return patterns;
}

/*
 * Get packets from a file. File format:
 * <dfa id> <num_bytes> byte_1 byte_2 ...
 * ...
 */
struct aho_pkt *aho_get_pkts(const char *pkt_file, int *num_pkts)
{
#if USE_RANDOM_PAYLOAD == 1
  uint64_t seed = 0xdeadbeef;
#endif

	assert(pkt_file != NULL && num_pkts != NULL);

	int i, j;
	struct aho_pkt *pkts;

	int dfa_load[AHO_MAX_DFA] = {0};

	FILE *pkt_fp = fopen(pkt_file, "r");
	assert(pkt_fp != NULL);

	/* Get the number of pkts and do a sanity check */
	fscanf(pkt_fp, "%d", num_pkts);
	if(*num_pkts > AHO_MAX_PKTS) {
		*num_pkts = AHO_MAX_PKTS;
	}
	printf("\taho: num_pkts = %d\n", *num_pkts);

	/* Initialize pkt pointers */
	pkts = (struct aho_pkt *) malloc(*num_pkts * 
		sizeof(struct aho_pkt));
	assert(pkts != NULL);
	memset(pkts, 0, *num_pkts * sizeof(struct aho_pkt));

	/* Get the actual packets */
	for(i = 0; i < *num_pkts; i++) {
		int dfa_id;
		int len;

		/* Give an identifier to this packet */
		pkts[i].pkt_id = i;

		/* Get the DFA ID of this packet */
		fscanf(pkt_fp, "%d", &dfa_id);
		assert(dfa_id >= 0 && dfa_id < AHO_MAX_DFA);
		pkts[i].dfa_id = dfa_id;

		/* Get the length of this packet */
		fscanf(pkt_fp, "%d", &len);
		pkts[i].len = len;		

		pkts[i].content = malloc(len);
		assert(pkts[i].content != NULL);

		/* Get one byte at a time */
		for(j = 0; j < len; j++) {
			int cur_byte;
			fscanf(pkt_fp, "%d", &cur_byte);
			assert(cur_byte >= 0 && cur_byte <= 255);
			pkts[i].content[j] = (uint8_t) cur_byte;

#if USE_RANDOM_PAYLOAD == 1
      uint64_t rand = fastrand(&seed);
			pkts[i].content[j] = (uint8_t) rand;
#endif
		}


		dfa_load[dfa_id]++;
		if(i % 10000 == 0) {
			printf("\t\taho: Read %d packets\n", i);
		}
	}

	printf("\taho: Printing DFAs with > 1000 packets\n");
	for(i = 0; i < AHO_MAX_DFA; i++) {
		if(dfa_load[i] > 1000) {
			printf("\t\taho: DFA %d has %d packets\n", i, dfa_load[i]);
		}
	}

	return pkts;
}

void aho_preprocess_dfa(struct aho_dfa *dfa)
{
	static int tot_states = 0;		/* Total states across all DFAs */
	tot_states += dfa->num_used_states;
	printf("\taho: Preprocessing DFA %d. num_states = %d | tot_states = %d\n",
		dfa->id, dfa->num_used_states, tot_states);
	assert(dfa != NULL);

	struct aho_state *st_arr = dfa->root;

	int i, j;
	for(i = 0; i <= dfa->num_used_states; i++) {

		/* Copy the matching states from the queue to an array
		 * We only have space for 16 matching patterns */
		assert(st_arr[i].output.count < 16);
		struct ds_qnode *t = st_arr[i].output.head;
		for(j = 0; j < st_arr[i].output.count; j++) {
			assert(t != NULL);
			st_arr[i].out_arr[j] = t->data;
			t = t->next;
		}

		for(j = 0; j < AHO_ALPHA_SIZE; j++) {
			if(st_arr[i].G[j] != AHO_FAIL) {
				continue;
			}

			int state_ij = i;
			while(st_arr[state_ij].G[j] == AHO_FAIL) {
				state_ij = st_arr[state_ij].F;
			}

			state_ij = st_arr[state_ij].G[j];
			st_arr[i].G[j] = state_ij;
		}
	}
}
