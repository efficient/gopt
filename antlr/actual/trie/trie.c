#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "trie.h"

inline trie_t *trie_init(void) 
{
    return malloc(sizeof(trie_t));
}

void trie_add(trie_t *t, char *word) 
{
	int c;
	while ((c = *word++)) {
		assert(c >= 0 && c < TRIE_SIZE);
		if (t->chars[c] == NULL) {
			t->chars[c] = trie_init();
		}
		t = t->chars[c];
	}
	t->sentinel = (void*) !NULL;
}

int trie_exists(trie_t *t, char *word) 
{
    int c;
    while ((c = *word++)) {
        if (t->chars[c] == NULL) {
            return 0;
        }
        t = t->chars[c];
    }
    return t->sentinel != NULL ? 1 : 0;
}

void trie_free(trie_t *t) 
{
	int i;
    for (i = /*skip sentinel*/ 1; i < TRIE_SIZE; i++) {
        if (t->chars[i] != NULL) {
            trie_free(t->chars[i]);
        }
    }
    free(t);
}
