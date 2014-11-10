#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>

#include "ds_queue.h"

/**< Maximum number of states, at least as large as the sum of all key word's length. */
#define MAX (505 * 505)

/**< Alphabet size, 26 for lower case English letters. */
#define ALPHA_SIZE 27
#define FAIL -1

/**< Maximum length of the text */
#define MAX_LENGTH 1000010

int g[MAX][ALPHA_SIZE];
int f[MAX];
struct ds_queue output[MAX];
char text[MAX_LENGTH];
int new_state;

void enter(char *word, int index)
{
	int length = strlen(word);
	int j, state = 0;
	for(j = 0; j < length; j++) {
		int c = word[j] - 'a';
		if(g[state][c] == FAIL) {
			break;
		}
		state = g[state][c];
	}

	/**< Characters j to length - 1 need new states */
	for(; j < length; j++) {
		int c = word[j] - 'a';
		new_state++;
		g[state][c] = new_state;
		state = new_state;
	}

	/**< Add this word as the output for the last state */
	assert(state >= 0 && state < MAX);
	ds_queue_add(&output[state], index);
}

int main(int argc, char *argv[])
{
	int n, i;
	char c;
	char word[505];
	int count[505];

	scanf("%d", &n);
	scanf("%s", text);

	new_state = 0;
	memset(g, FAIL, sizeof g);
	memset(f, FAIL, sizeof f);

	/**< Initialize output queues for each state */
	for(i = 0; i < MAX; i++) {
		ds_queue_init(&output[i]);
	}

	/**< Read patterns and build the Trie */
	for(i = 0; i < n; i++) {
		scanf("%s", word);
		count[i] = 0;
		enter(word, i);
	}

	/**< Invalid transitions from the root state need to loop back */
	for(c = 'a'; c <= 'z'; c++) {
		int a = c - 'a';
		if(g[0][a] == FAIL) {
			g[0][a] = 0;
		}
	}

	/**< Build failure function: add root node's children to the queue */
	struct ds_queue Q;
	ds_queue_init(&Q);
	for(c = 'a'; c <= 'z'; c++) {
		int a = c - 'a';
		int s = g[0][a];
		if(s != 0) {
			ds_queue_add(&Q, s);
			f[s] = 0;
		}
	}

	while(!ds_queue_is_empty(&Q)) {
		int r = ds_queue_remove(&Q);

		/**< Look at all the valid state transitions from r */
		for(c = 'a'; c <= 'z'; c++) {
			int a = c - 'a';
			int s = g[r][a];

			if(s != FAIL) {
				ds_queue_add(&Q, s);
				int state = f[r];
				while(g[state][a] == FAIL) {
					state = f[state];
				}
				f[s] = g[state][a];

				/**< Add all patterns from output[f[s]] to output[s] */
				struct ds_qnode *t = output[f[s]].head;
				while(t != NULL) {
					ds_queue_add(&output[s], t->data);
					t = t->next;
				}
			}
		}
	}

	/**< Count occurrences of patterns inside text */
	int state = 0;
	int length = strlen(text);
	for(i = 0; i < length; i++) {
		int a = text[i] - 'a';
		while(g[state][a] == FAIL) {
			state = f[state];
		}
		state = g[state][a];

		/**< Increase the count for all patterns matched at this state */
		struct ds_qnode *t = output[state].head;
		while(t != NULL) {
			count[t->data] ++;
			t = t->next;
		}
	}

	for(i = 0; i < n; i++) {
		printf("%d\n", count[i]);
	}

	return 0;
}
