#define TRIE_SIZE 128

typedef struct trie_t {
    void *sentinel;
    struct trie_t *chars[TRIE_SIZE];
} trie_t;

trie_t *trie_init(void);
void trie_add(trie_t *, char *);
int trie_exists(trie_t *, char *);
int trie_load(trie_t *, char *);
void trie_free(trie_t *);
