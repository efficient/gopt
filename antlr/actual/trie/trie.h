#ifndef _TRIE_H_
#define _TRIE_H_

#define TRIE_SIZE 128

typedef struct trie_t {
    void *sentinel;
    struct trie_t *chars[TRIE_SIZE];
} trie_t;

trie_t *trie_init(void);
void trie_add(trie_t *, char *);
int trie_exists(trie_t *, char *);
int trie_load(trie_t *, char *);
void trie_strip(trie_t *, char *, char *);
void trie_free(trie_t *);

#define trie_step(t,c) (t = (t == NULL || t->chars[c] == NULL ? NULL : t->chars[c]))
#define trie_word(t) (t != NULL && t->sentinel != NULL)

#endif
