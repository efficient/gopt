example: trie.h trie.c example.c
	@$(CC) -Wall -pedantic -ansi -std=c99 -g -O3 -o example trie.c example.c

clean:
	@rm -f ./example

run: example
	@./example

all: clean example run
