#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>

#include "aho.h"
#include "util.h"

int main(int argc, char *argv[])
{
	int n, i;
	char *text, *pattern;
	int *count;

	struct aho_state *dfa;
	aho_init(&dfa);
	
	scanf("%d", &n);
	scanf("%ms", &text);

	/**< Initialize the pattern-occurence count */
	count = malloc(n * sizeof(int));
	memset(count, 0, n * sizeof(int));

	/**< Read patterns and build the Trie */
	red_printf("Building AC goto function\n");
	for(i = 0; i < n; i++) {
		scanf("%ms", &pattern);
		count[i] = 0;
		aho_add_pattern(dfa, pattern, i);
	}

	/**< Create the failure function */
	red_printf("Building AC failure function\n");
	aho_build_ff(dfa);

	/**< Count occurrences of patterns inside text */
	red_printf("Starting lookup\n");
	int state = 0;
	int length = strlen(text);
	for(i = 0; i < length; i++) {
		int inp = text[i];
		while(dfa[state].G[inp] == AHO_FAIL) {
			state = dfa[state].F;
		}
		state = dfa[state].G[inp];

		/**< Increase the count for all patterns matched at this state */
		struct ds_qnode *t = dfa[state].output.head;
		while(t != NULL) {
			count[t->data] ++;
			t = t->next;
		}
	}

	for(i = 0; i < n; i++) {
		int expected;
		scanf("%d\n", &expected);
		if(count[i] != expected) {
			red_printf("Failed. count[%d] = %d, expected = %d\n",
				i, count[i], expected);
			exit(-1);
		}
	}

	red_printf("Passed!\n");

	free(pattern);
	free(text);
	free(count);
	free(dfa);

	return 0;
}
