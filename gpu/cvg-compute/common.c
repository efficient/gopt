#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include "common.h"

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

/**< The log used in random memory rate tests is a random permutation */
void init_ht_log(int *log, int n)
{
	printf("\tCreating a rand permutation of 0-%d. This takes time..\n", n);
	fflush(stdout);

	int i, j;
	assert(log != NULL && n >= 1);

	for(i = 0; i < n; i ++) {
		log[i] = i;
	}

	/**< Shuffle to create a random permutation */
	for(i = n - 1; i >= 1; i --) {
		j = rand() % (i + 1);
		int temp = log[i];
		log[i] = log[j];
		log[j] = temp;
	}
}
