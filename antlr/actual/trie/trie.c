#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "trie.h"

inline trie_t *trie_init(void) {
    return calloc(1, sizeof(trie_t));
}

void trie_add(trie_t *t, char *word) {
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

int trie_exists(trie_t *t, char *word) {
    int c;
    while ((c = *word++)) {
        if (t->chars[c] == NULL) {
            return 0;
        }
        t = t->chars[c];
    }
    return t->sentinel != NULL ? 1 : 0;
}

int trie_load(trie_t *t, char *file) {
    FILE *stream = fopen(file, "r");
    if (stream == NULL) {
        return -1;
    }

    trie_t *root = t;
    int c, words = 0, word_len = 0;
    while ((c = getc(stream)) != EOF) {
        if (c == '\n' || c == '\r') {
            if (word_len > 0) {
                t->sentinel = (void*) !NULL;
                words++;
                word_len = 0;
                t = root;
            }
        } else {
            word_len++;
            assert(c >= 0 && c < TRIE_SIZE);
            if (t->chars[c] == NULL) {
                t->chars[c] = trie_init();
            }
            t = t->chars[c];
        }
    }
    fclose(stream);
    if (t != root && word_len > 0) {
        t->sentinel = (void*) !NULL;
    }
    return words;
}

void trie_strip(trie_t *t, char *src, char *dest) {
    if (src == NULL) {
        return;
    }
    if (dest == NULL) {
        dest = src;
    }
    int c, i = 0, last_break = 0, in_trie = 1;
    trie_t *root = t;

    while ((c = dest[i++] = *src++)) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            t = root;
            if (in_trie) {
                i = last_break;
            } else {
                in_trie = 1;
                last_break = i;
            }
            continue;
        }
        if (!in_trie) {
            continue;
        }
        if (t->chars[c] == NULL) {
            in_trie = 0;
        } else {
            t = t->chars[c];
            in_trie = 1;
        }
    }
}

void trie_free(trie_t *t) {
    for (int i = /*skip sentinel*/ 1; i < TRIE_SIZE; i++) {
        if (t->chars[i] != NULL) {
            trie_free(t->chars[i]);
        }
    }
    free(t);
}
