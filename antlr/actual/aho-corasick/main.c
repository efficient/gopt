#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<unistd.h>

#include "aho.h"
#include "util.h"

int main(int argc, char *argv[])
{
	int num_patterns, i;
	char **patterns, *first_newline = NULL;
	int *count;
	size_t buf_size;

	struct aho_state *dfa;
	aho_init(&dfa);
	
	/**< Get the number of patterns and do a sanity check */
	scanf("%d", &num_patterns);
	assert(num_patterns >=0 && num_patterns <= AHO_MAX_PATTERNS);
	printf("num_patterns = %d\n", num_patterns);

	/**< Get the newline after num_patterns */
	getline(&first_newline, &buf_size, stdin);

	/**< Initialize pattern pointers: input to getline() should ne NULL */
	patterns = (char **) malloc(num_patterns * sizeof(char *));
	memset(patterns, 0, num_patterns * sizeof(char *));
	assert(patterns != NULL);

	/**< Read patterns and build the Trie */
	red_printf("Building AC goto function: \n");
	for(i = 0; i < num_patterns; i ++) {
		int num_chars = getline(&patterns[i], (size_t *) &buf_size, stdin);

		/**< Zero out the newline at the end of getline()'s output*/
		patterns[i][num_chars - 1] = 0;

		aho_add_pattern(dfa, patterns[i], i);
		printf("Adding pattern %d: %s\n", i, patterns[i]);
		usleep(20000);
	}

	/**< Create the failure function */
	red_printf("Building AC failure function\n");
	aho_build_ff(dfa);

	/**< Count occurrences of patterns inside text */
	red_printf("Starting lookups\n");
	int state = 0, final_state_sum = 0;

	for(i = 0; i < num_patterns; i ++) {

		int state = 0;
		int pattern_len = strlen(patterns[i]), j;

		for(j = 0; j < pattern_len; j ++) {
			int inp = patterns[i][j];
			while(dfa[state].G[inp] == AHO_FAIL) {
				state = dfa[state].F;
			}

			state = dfa[state].G[inp];
		}

		final_state_sum += state;

	}

	red_printf("Final state sum = %d\n", final_state_sum);

	for(i = 0; i < num_patterns; i ++) {
		free(patterns[i]);
	}

	free(patterns);
	free(dfa);

	return 0;
}
