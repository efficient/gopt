#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trie.h"

int main(int argc, char **argv) {
    trie_t *t = trie_init();

    //Load the words from stopwords.txt into a trie
    trie_load(t, "stopwords.txt");

    printf("'foo': %s\n", trie_exists(t, "foo") ? "yes" : "no");
    printf("'among': %s\n", trie_exists(t, "among") ? "yes" : "no");

    trie_free(t);

    return 0;
}
