#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <papi.h>

#include "trie.h"

#define NUM_WORDS 235885
#define MAX_WORD_LEN 26		// Max word length in dictionary

#define NUM_LOOKUPS 10000000

inline uint32_t fastrand(uint64_t* seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}

int main(int argc, char **argv) 
{
	int i, retval;
	uint64_t seed = 0xdeadbeef;

	// Variables for PAPI
	float real_time, proc_time, ipc;
	long long ins;

	trie_t *t = trie_init();

	// Get the words into memory and add them to the trie
	char **words = malloc(NUM_WORDS * sizeof(void *));
	for(i = 0; i < NUM_WORDS; i ++) {
		words[i] = malloc(MAX_WORD_LEN * sizeof(char));
	}

	for(i = 0; i < NUM_WORDS; i ++) {
		scanf("%s", words[i]);
		trie_add(t, words[i]);
	}
	
	red_printf("Done adding words to trie\n");

	// Init PAPI_TOT_INS and PAPI_TOT_CYC counters
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("retval: %d\n", retval);
		exit(1);
	}

	// Do some lookups
	int num_exists = 0;
	for(i = 0; i < NUM_LOOKUPS; i ++) {
		int index = fastrand(&seed) & 131071;
		num_exists += trie_exists(t, words[index]);
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("retval: %d\n", retval);
		exit(1);
	}

	printf("num_exists = %d\n", num_exists);

	red_printf("Real_time: %fs, Total instructions: %lld, Total cycles = %lld, IPC: %f\n", 
		real_time, ins, (long long ) (ins / ipc), ipc);

	// Be a good boy and free
	for(i = 0; i < NUM_WORDS; i ++) {
		free(words[i]);
	}
	free(words);

	trie_free(t);

	return 0;
}
