#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <assert.h>

#include "ipv4_rtable.h"
#include "cpu_ticks.h"
#include "utility.h"

unsigned n = 1 << 24;
struct ipv4_rib_entry *rib_entries;

unsigned num_addrs = 100000;

/* unsigned num_burst_sizes = 16; */
/* unsigned burst_size_array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; */
unsigned num_burst_sizes = 4;
unsigned burst_size_array[] = {4, 8, 12, 16};

double naive_rate[32], mem_rate[32];

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
	printf("sizeof ipv4_entry = %lu\n", sizeof(struct ipv4_rtable_entry));

	unsigned i, j, k;

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

	struct timeval start, end;

	for (i = 0; i < num_burst_sizes; i++) {
		printf("batch-size: %d", burst_size_array[i]);

		// Basic lookups
		gettimeofday(&start, 0);
		for (j = 0; j < num_addrs; j ++) {
			naive_port_id_array[j] = ipv4_rtable_lookup(table, addr_array[j]);
		}
		gettimeofday(&end, 0);
		naive_rate[i] = (double) num_addrs / time_elapsed(&start, &end);

		// Batched lookups
		gettimeofday(&start, 0);
		for (j = 0; j < num_addrs; j += burst_size_array[i]) {
			ipv4_rtable_lookup_multi(table, burst_size_array[i], addr_array + j, mem_port_id_array + j);
		}
		gettimeofday(&end, 0);
		mem_rate[i] = (double) num_addrs / time_elapsed(&start, &end);

		// Compare output
		for (k = 0; k < burst_size_array[i]; k ++) {
			assert(naive_port_id_array[k] == mem_port_id_array[k]);
		}

		printf("\tdone\n");
	}

	ipv4_rtable_print_statistics();

	/* printf("                 cpu                  memory-optimized\n"); */
	for (i = 0; i < num_burst_sizes; i++) {
		printf("size %2d packets, rate %8.3lf Mpps", burst_size_array[i], naive_rate[i]);
		printf("      %8.3lf Mpps\n", mem_rate[i]);
	}

	return 0;
}
