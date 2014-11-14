#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<unistd.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#include "aho.h"

static int aho_new_state = 0;	/**< The last used state */

/**< Initialize the state transition table and o/p queues */
void aho_init(struct aho_state **dfa)
{
	int i;
	int sid = shmget(AHO_SHM_KEY, AHO_MAX_STATES * sizeof(struct aho_state),
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("\tCould not allocate DFA states for AC\n");
		exit(-1);
	}

	*dfa = shmat(sid, 0, 0);
	assert(*dfa != NULL);

	for(i = 0; i < AHO_MAX_STATES; i ++) {
		struct aho_state *cur_state = &((*dfa)[i]);
		memset(cur_state->G, AHO_FAIL, AHO_ALPHA_SIZE * sizeof(int));
		cur_state->F = AHO_FAIL;
		ds_queue_init(&cur_state->output);
	}
}

/**< Add a pattern to the DFA */
void
aho_add_pattern(struct aho_state *dfa, struct aho_pattern *pattern, int index)
{
	int length = pattern->len;
	assert(length >= 0 && length <= 100000);
	int j, state = 0;

	for(j = 0; j < length; j ++) {
		if(dfa[state].G[pattern->content[j]] == AHO_FAIL) {
			break;
		}
		state = dfa[state].G[pattern->content[j]];
	}

	/**< Characters j to (length - 1) need new states */
	for(; j < length; j ++) {
		aho_new_state ++;

		/**< AHO_MAX_STATES - 1 is AHO_FAIL */
		assert(aho_new_state <= AHO_MAX_STATES - 2);

		/**< Print when states consume around 30 MB */
		if(aho_new_state % 10000 == 0) {
			printf("\taho: number of states = %d\n", aho_new_state);
		}

		dfa[state].G[pattern->content[j]] = aho_new_state;
		state = aho_new_state;
	}

	/**< Add this pattern as the output for the last state */
	if(state == 0) {
		printf("Error for index = %d\n", index);
		exit(-1);
	}
	ds_queue_add(&(dfa[state].output), index);
}

/**< Build the Aho-Coraick failure function */
void aho_build_ff(struct aho_state *dfa)
{
	int i;
	struct ds_queue state_queue;
	ds_queue_init(&state_queue);

	/**< Invalid transitions from the root state need to loop back */
	for(i = 0; i < AHO_ALPHA_SIZE; i ++) {
		if(dfa[0].G[i] == AHO_FAIL) {
			dfa[0].G[i] = 0;
		}
	}

	/**< Initialize the failure function of the root's children */
	for(i = 0; i < AHO_ALPHA_SIZE; i ++) {
		int next_state = dfa[0].G[i];
		if(next_state != 0) {
			ds_queue_add(&state_queue, next_state);
			dfa[next_state].F = 0;
		}
	}

	/**< Create the failure function recursively */
	while(!ds_queue_is_empty(&state_queue)) {
		int cur_state = ds_queue_remove(&state_queue);

		/**< Look at all the valid state transitions from cur_state */
		for(i = 0; i < AHO_ALPHA_SIZE; i ++) {
			int child_state = dfa[cur_state].G[i];

			if(child_state != AHO_FAIL) {
				ds_queue_add(&state_queue, child_state);

				int best_state = dfa[cur_state].F;
				while(dfa[best_state].G[i] == AHO_FAIL) {
					best_state = dfa[best_state].F;
				}

				int child_fail_state = dfa[best_state].G[i];
				dfa[child_state].F = child_fail_state;

				/**< Add all patterns from child_state.F to child_state */
				struct ds_qnode *t = dfa[child_fail_state].output.head;
				while(t != NULL) {
					ds_queue_add(&dfa[child_state].output, t->data);
					t = t->next;
				}
			}
		}	/**< End loop over children states */
	}	/**< Finish traversal */
}

/**< Get variable length strings from a pattern file */
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

	/**< Get the number of patterns and do a sanity check */
	fscanf(pattern_fp, "%d", num_patterns);
	assert(*num_patterns >= 0 && *num_patterns <= AHO_MAX_PATTERNS);
	printf("\taho: num_patterns = %d\n", *num_patterns);

	/**< Get the newline after num_patterns (otherwise getline() reads it) */
	getline(&first_newline, &buf_size, pattern_fp);

	/**< Initialize pattern pointers: input to getline() should ne NULL */
	patterns = (struct aho_pattern *) malloc(*num_patterns * 
		sizeof(struct aho_pattern));
	assert(patterns != NULL);
	memset(patterns, 0, *num_patterns * sizeof(struct aho_pattern));

	/**< Read the actual patterns from */
	for(i = 0; i < *num_patterns; i ++) {
		int num_chars = getline((char **) &patterns[i].content, 
			(size_t *) &buf_size, pattern_fp);

		patterns[i].len = num_chars - 1;
		assert(patterns[i].len <= AHO_MAX_PATTERN_LEN);

		/**< Zero out the newline at the end */
		patterns[i].content[num_chars - 1] = 0;
	}

	return patterns;
}

/**< Get Snort's content strings from a byte-file. File format:
  *  <num_contents>
  *  <num_bytes> byte_1 byte_2 ...
  *  ...
  */
struct aho_pattern 
*aho_get_patterns(const char *pattern_file, int *num_patterns)
{
	assert(pattern_file != NULL && num_patterns != NULL);

	int i, j;
	struct aho_pattern *patterns;
	size_t buf_size;

	FILE *pattern_fp = fopen(pattern_file, "r");
	assert(pattern_fp != NULL);

	/**< Get the number of patterns and do a sanity check */
	fscanf(pattern_fp, "%d", num_patterns);
	assert(*num_patterns >= 0 && *num_patterns <= AHO_MAX_PATTERNS);
	printf("\taho: num_patterns = %d\n", *num_patterns);

	/**< Initialize pattern pointers */
	patterns = (struct aho_pattern *) malloc(*num_patterns * 
		sizeof(struct aho_pattern));
	assert(patterns != NULL);
	memset(patterns, 0, *num_patterns * sizeof(struct aho_pattern));

	/**< Get the actual content strings */
	for(i = 0; i < *num_patterns; i ++) {
		int num_bytes;
		fscanf(pattern_fp, "%d", &num_bytes);
		assert(num_bytes >= 0 && num_bytes <= AHO_MAX_PATTERN_LEN);

		patterns[i].len = num_bytes;
		patterns[i].content = malloc(num_bytes);
		assert(patterns[i].content != NULL);

		/**< Get one byte at a time */
		for(j = 0; j < num_bytes; j ++) {
			int cur_byte;
			fscanf(pattern_fp, "%d\n", &cur_byte);
			assert(cur_byte >= 0 && cur_byte <= 255);
			patterns[i].content[j] = (uint8_t) cur_byte;
		}
	}

	return patterns;
}

void aho_preprocess_dfa(struct aho_state *dfa)
{
	int i, j;
	for(i = 0; i <= aho_new_state; i ++) {
		for(j = 0; j < AHO_ALPHA_SIZE; j ++) {
			if(dfa[i].G[j] != AHO_FAIL) {
				continue;
			}

			int state_ij = i;
			while(dfa[state_ij].G[j] == AHO_FAIL) {
				state_ij = dfa[state_ij].F;
			}

			state_ij = dfa[state_ij].G[j];
			dfa[i].G[j] = state_ij;
		}
	}
}
