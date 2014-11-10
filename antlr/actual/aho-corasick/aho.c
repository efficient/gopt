#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>

#include "aho.h"

/**< Initialize the state transition table and state outputs */
void aho_init(struct aho_state **dfa)
{
	int i;
	*dfa = malloc(AHO_MAX_STATES * sizeof(struct aho_state));
	for(i = 0; i < AHO_MAX_STATES; i ++) {
		struct aho_state *cur_state = &((*dfa)[i]);
		memset(cur_state->G, AHO_FAIL, AHO_ALPHA_SIZE * sizeof(int));
		cur_state->F = AHO_FAIL;
		ds_queue_init(&cur_state->output);
	}
}

/**< Add a pattern to the DFA */
void aho_add_pattern(struct aho_state *dfa, char *pattern, int index)
{
	static int aho_new_state = 0;

	int length = strlen(pattern);
	int j, state = 0;

	for(j = 0; j < length; j ++) {
		if(dfa[state].G[pattern[j]] == AHO_FAIL) {
			break;
		}
		state = dfa[state].G[pattern[j]];
	}

	/**< Characters j to (length - 1) need new states */
	for(; j < length; j ++) {
		aho_new_state ++;
		assert(aho_new_state < AHO_MAX_STATES);

		dfa[state].G[pattern[j]] = aho_new_state;
		state = aho_new_state;
	}

	/**< Add this pattern as the output for the last state */
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

