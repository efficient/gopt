#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <assert.h>

#include "ipv4_rtable.h"
#include "cpu_ticks.h"
#include "utility.h"
#include "fpp.h"

unsigned n = 1 << 24;
//unsigned n = 1 << 16;		// L2 cache version
struct ipv4_rib_entry *rib_entries;

unsigned num_addrs = (4 * 1024 * 1024);

void
generate_rib_entries(struct ipv4_rib_entry **rib_entries_ptr, unsigned n)
{
    unsigned i, j;

    struct ipv4_rib_entry *rib_entries = (struct ipv4_rib_entry *) malloc(sizeof(struct ipv4_rib_entry) * n);

    for (i = 0; i < n; i++) {
        int x = rand() % 3 + 1;
        uint32_t addr = 0;

        rib_entries[i].netmask_num_bits = x * 8;
        rib_entries[i].netmask = ((uint32_t) 0xFFFFFFFF) << (32 - x * 8);

        switch (x) {
        case 3:
            addr |= (rand() % 256) << 8;
        case 2:
            addr |= (rand() % 256) << 16;
        case 1:
            addr |= (rand() % 256) << 24;
        }
        
        rib_entries[i].addr = addr;
        rib_entries[i].port_id = (uint8_t) rand() % 127;
    }

    *rib_entries_ptr = rib_entries;
}

int main(int argc, char **argv)
{
	unsigned i, j, k;
	double naive_rate, mem_rate, nogoto_rate, goto_rate;

	printf("sizeof ipv4_entry = %lu\n", sizeof(struct ipv4_rtable_entry));

	generate_rib_entries(&rib_entries, n);

	struct ipv4_rtable *table = ipv4_rtable_create(rib_entries, n, 0);

	free(rib_entries);

	uint32_t *addr_array = (uint32_t *) malloc(sizeof(uint32_t) * num_addrs);
	for (i = 0; i < num_addrs; i++) {
		uint32_t addr = 0;
		for (k = 0; k < 4; k++) {
			addr = (addr << 8) | (rand() % 256);
		}
		addr_array[i] = addr;
	}

	uint8_t *naive_port_id_array = (uint8_t *) malloc(sizeof(uint8_t) * num_addrs);
	uint8_t *mem_port_id_array = (uint8_t *) malloc(sizeof(uint8_t) * num_addrs);
	uint8_t *nogoto_port_id_array = (uint8_t *) malloc(sizeof(uint8_t) * num_addrs);
	uint8_t *goto_port_id_array = (uint8_t *) malloc(sizeof(uint8_t) * num_addrs);

	struct timeval start, end;

	printf("Batch size = %d\n", BATCH_SIZE);

	// Basic lookups
	gettimeofday(&start, 0);
	for (j = 0; j < num_addrs; j ++) {
		naive_port_id_array[j] = ipv4_rtable_lookup(table, addr_array[j]);
	}
	gettimeofday(&end, 0);
	naive_rate = (double) num_addrs / time_elapsed(&start, &end);

	// Batched lookups
	gettimeofday(&start, 0);
	for (j = 0; j < num_addrs; j += BATCH_SIZE) {
		ipv4_rtable_lookup_multi(table, addr_array + j, mem_port_id_array + j);
	}
	gettimeofday(&end, 0);
	mem_rate = (double) num_addrs / time_elapsed(&start, &end);

	// Nogoto lookups
	gettimeofday(&start, 0);
	for (j = 0; j < num_addrs; j += BATCH_SIZE) {
		ipv4_rtable_lookup_nogoto(table, addr_array + j, nogoto_port_id_array + j);
	}
	gettimeofday(&end, 0);
	nogoto_rate = (double) num_addrs / time_elapsed(&start, &end);
	
	// goto lookups
	gettimeofday(&start, 0);
	for (j = 0; j < num_addrs; j += BATCH_SIZE) {
		ipv4_rtable_lookup_goto(table, addr_array + j, goto_port_id_array + j);
	}
	gettimeofday(&end, 0);
	goto_rate = (double) num_addrs / time_elapsed(&start, &end);

	// Compare output
	for (k = 0; k < num_addrs; k ++) {
		assert(naive_port_id_array[k] == mem_port_id_array[k]);
		assert(nogoto_port_id_array[k] == mem_port_id_array[k]);
		assert(goto_port_id_array[k] == mem_port_id_array[k]);
	}

	printf("\tDone\n");

	ipv4_rtable_print_statistics();

	/* printf("                 cpu                  memory-optimized\n"); */
	printf("Batch size:\t %d\n\n", BATCH_SIZE);
	printf("\tNaive rate:\t %8.3lf Mpps\n", naive_rate);
	printf("\tHandopt rate:\t %8.3lf Mpps\n", mem_rate);
	printf("\tNogoto rate:\t %8.3lf Mpps\n", nogoto_rate);
	printf("\tGoto rate:\t %8.3lf Mpps\n", goto_rate);

	return 0;
}
