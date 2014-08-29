#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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

// Like printf, but red. Limited to 1000 characters.
void red_printf(const char *format, ...)
{	
	#define RED_LIM 1000
	va_list args;
	int i;

	char buf1[RED_LIM], buf2[RED_LIM];
	memset(buf1, 0, RED_LIM);
	memset(buf2, 0, RED_LIM);

    va_start(args, format);

	// Marshal the stuff to print in a buffer
	vsnprintf(buf1, RED_LIM, format, args);

	// Probably a bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf1[i] == 0);
	}

	// Add markers for red color and reset color
	snprintf(buf2, 1000, "\033[31m%s\033[0m", buf1);

	// Probably another bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf2[i] == 0);
	}

	printf("%s", buf2);

    va_end(args);
}

