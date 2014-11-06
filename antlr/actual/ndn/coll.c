#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <assert.h>

#include "fpp.h"
#include "util.h"
#include "ndn.h"

struct ndn_bucket *ht_index;

int main(void)
{
	struct ndn_ht ht;

	red_printf("Creating cuckoo index..\n");
	ndn_init("/home/akalia/fastpp/data_dump/ndn_distributed_sample",
		0xf, &ht);
	red_printf("\tSetting up cuckoo index done!\n");

	/**< Check if all the prefixes were inserted successfully */
	red_printf("Looking up all URLs\n");
	ndn_check("/home/akalia/fastpp/data_dump/ndn_distributed_sample", &ht);

	return 0;
}

