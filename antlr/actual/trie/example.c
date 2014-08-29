#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trie.h"

#define NUM_WORDS 235885
#define MAX_WORD_LEN 26

int main(int argc, char **argv) 
{
	int i;
	trie_t *t = trie_init();

	char **words = malloc(NUM_WORDS * sizeof(void *));
	for(i = 0; i < NUM_WORDS; i ++) {
		words[i] = malloc(MAX_WORD_LEN * sizeof(char));
	}
	
	for(i = 0; i < NUM_WORDS; i ++) {
		scanf("%s", words[i]);
		trie_add(t, words[i]);
	}

	printf("'foo': %s\n", trie_exists(t, "foo") ? "yes" : "no");
	printf("'among': %s\n", trie_exists(t, "among") ? "yes" : "no");

	for(i = 0; i < NUM_WORDS; i ++) {
		free(words[i]);
	}
	free(words);

	trie_free(t);

	return 0;
}
