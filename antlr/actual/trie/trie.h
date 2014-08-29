#define TRIE_SIZE 128

typedef struct trie_t {
    void *sentinel;
    struct trie_t *chars[TRIE_SIZE];
} trie_t;

void trie_add(trie_t *t, char *word);
trie_t *trie_init(void);
int trie_exists(trie_t *, char *);
void trie_free(trie_t *);

void red_printf(const char *format, ...);
