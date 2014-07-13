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

struct testsuit {
    struct ipv4_rtable *table;
    uint32_t *addr_array;
    uint8_t *naive_port_id_array;
    uint8_t *mem_port_id_array;
};

unsigned num_addrs = 100000;
/* unsigned num_burst_sizes = 16; */
/* unsigned burst_size_array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; */
unsigned num_burst_sizes = 4;
unsigned burst_size_array[] = {4, 8, 12, 16};
unsigned num_testsuits = 3;
struct testsuit testsuits[10];
double naive_rate[32], mem_rate[32];

void
usage(char *program)
{
    printf("usage: %s\n", program);
}

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
    uint32_t addr;

    srand(time(0));

    for (i = 0; i < num_testsuits; i++) {
        generate_rib_entries(&rib_entries, n);

        struct ipv4_rtable *table = ipv4_rtable_create(rib_entries, n, 0);

        testsuits[i].table = table;
        assert(testsuits[i].table);

        free(rib_entries);

        testsuits[i].addr_array = (uint32_t *) malloc(sizeof(uint32_t) * num_addrs);
        for (j = 0; j < num_addrs; j++) {
            addr = 0;
            for (k = 0; k < 4; k++)
                addr = (addr << 8) | (rand() % 256);
            testsuits[i].addr_array[j] = addr;
        }
        testsuits[i].naive_port_id_array = (uint8_t *) malloc(sizeof(uint8_t) * num_addrs);
        testsuits[i].mem_port_id_array = (uint8_t *) malloc(sizeof(uint8_t) * num_addrs);
    }

    unsigned test, num_tests = 10;
    struct timeval start, end;

    for (i = 0; i < num_burst_sizes; i++) {
        printf("batch-size: %d", burst_size_array[i]);
        
        gettimeofday(&start, 0);
        for (test = 0; test < num_tests; test++) {
            for (j = 0; j < num_testsuits; j++)
                for (k = 0; k < num_addrs; k++)
                    testsuits[j].naive_port_id_array[k] = ipv4_rtable_lookup(testsuits[j].table, testsuits[j].addr_array[k]);
        }
        gettimeofday(&end, 0);
        naive_rate[i] = (double)(num_addrs * num_tests * num_testsuits) / time_elapsed(&start, &end);

        gettimeofday(&start, 0);
        for (test = 0; test < num_tests; test++) {
            for (j = 0; j < num_testsuits; j++)
                for (k = 0; k < num_addrs; k += burst_size_array[i])
                    ipv4_rtable_lookup_multi(testsuits[j].table, burst_size_array[i], testsuits[j].addr_array + k, testsuits[j].mem_port_id_array + k);
        }
        gettimeofday(&end, 0);
        mem_rate[i] = (double)((num_addrs / burst_size_array[i]) * burst_size_array[i] * num_tests * num_testsuits) / time_elapsed(&start, &end);

        for (j = 0; j < num_testsuits; j++)
            for (k = 0; k < burst_size_array[i]; k++) {
                assert(testsuits[j].naive_port_id_array[k] == testsuits[j].mem_port_id_array[k]);
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
