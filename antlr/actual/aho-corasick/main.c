#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<unistd.h>
#include<papi.h>

#include "aho.h"
#include "util.h"

#define PATTERN_FILE "/home/akalia/fastpp/data_dump/snort_content_strings"

int main(int argc, char *argv[])
{
	int num_patterns, i;
	int *count;

	struct aho_state *dfa;
	aho_init(&dfa);

	struct aho_pattern *patterns = aho_get_patterns(PATTERN_FILE, 
		&num_patterns);

	red_printf("Building AC goto function: \n");
	for(i = 0; i < num_patterns; i ++) {
		aho_add_pattern(dfa, patterns[i].content, i);
	}

	/**< Create the failure function */
	red_printf("Building AC failure function\n");
	aho_build_ff(dfa);

	red_printf("Starting lookups\n");
	int final_state_sum = 0;

	/** < Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	/** < Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	for(i = 0; i < num_patterns; i ++) {
		int state = 0;
		int pattern_len = patterns[i].len, j;

		for(j = 0; j < pattern_len; j ++) {
			int inp = patterns[i].content[j];
			while(dfa[state].G[inp] == AHO_FAIL) {
				state = dfa[state].F;
			}

			state = dfa[state].G[inp];
		}

		final_state_sum += state;
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	red_printf("Time = %.4f s, Instructions = %lld, IPC = %f, sum = %d\n",
		real_time, ins, ipc, final_state_sum);

	for(i = 0; i < num_patterns; i ++) {
		free(patterns[i].content);
	}

	free(patterns);
	free(dfa);

	return 0;
}
