#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <xmmintrin.h>

#include "ipv4_rtable.h"
#include "utility.h"
#include "fpp.h"

struct ipv4_rtable_writeable_entry {
    unsigned entry_id;
    uint8_t port_id;
    struct ipv4_rtable_writeable_entry *children[1 << IPV4_RTABLE_ENTRY_NUM_BITS];
    struct ipv4_rtable_writeable_entry *prev;
    struct ipv4_rtable_writeable_entry *next;
};

static struct ipv4_rtable_writeable_entry *head;
static uint64_t total_memory_accesses = 0;
static uint64_t total_queries = 0;

static struct ipv4_rtable_writeable_entry *
ipv4_rtable_writeable_entry_alloc(unsigned *entry_id, uint8_t port_id)
{
    struct ipv4_rtable_writeable_entry *entry = (struct ipv4_rtable_writeable_entry *) malloc(sizeof(struct ipv4_rtable_writeable_entry));

    memset(entry, 0, sizeof(struct ipv4_rtable_writeable_entry));
    entry->entry_id = (*entry_id)++;
    entry->port_id = port_id;
    entry->prev = entry->next = NULL;

    return entry;
}

static void
append_entry(struct ipv4_rtable_writeable_entry *entry)
{
    if (!head)
        head = entry;
    else {
        entry->next = head;
        head->prev = entry;
        head = entry;
    }
}

static void
dispose_entry()
{
    struct ipv4_rtable_writeable_entry *entry = head;

    while (entry) {
        struct ipv4_rtable_writeable_entry *next_entry = entry->next;
        free(entry);
        entry = next_entry;
    }
}

struct ipv4_rtable *
ipv4_rtable_create(struct ipv4_rib_entry *rib_entries, unsigned n, uint8_t fallback_port_id)
{
    unsigned i;
    int shift;
    unsigned num_entries = 0;
    struct ipv4_rtable_writeable_entry *root_entry, *entry;

    head = NULL;
    root_entry = ipv4_rtable_writeable_entry_alloc(&num_entries, fallback_port_id);
    append_entry(root_entry);

    for (i = 0; i < n; i++) {
        for (entry = root_entry, shift = 32 - IPV4_RTABLE_ENTRY_NUM_BITS; shift >= 32 - rib_entries[i].netmask_num_bits; shift -= IPV4_RTABLE_ENTRY_NUM_BITS) {
            uint32_t x = (rib_entries[i].addr >> shift) & (~((~(uint32_t) 0) << IPV4_RTABLE_ENTRY_NUM_BITS));

            if (!entry->children[x]) {
                struct ipv4_rtable_writeable_entry *child_entry = ipv4_rtable_writeable_entry_alloc(&num_entries, fallback_port_id);
                append_entry(child_entry);
                entry->children[x] = child_entry;
            }
            entry = entry->children[x];
        }

        entry->port_id = rib_entries[i].port_id;
    }

    struct ipv4_rtable *rtable = (struct ipv4_rtable *) malloc(sizeof(struct ipv4_rtable) + num_entries * sizeof(struct ipv4_rtable_entry));
    rtable->entries = (struct ipv4_rtable_entry *) ((char *) rtable + sizeof(struct ipv4_rtable));
    struct ipv4_rtable_entry *rtable_entries = (struct ipv4_rtable_entry *) rtable->entries;

    printf("num_entries: %u\n", num_entries);
    printf("memory: %.2lf MB\n", (double) (sizeof(struct ipv4_rtable) + num_entries * sizeof(struct ipv4_rtable_entry)) / 1024 / 1024);

    rtable->n = num_entries;
    rtable->fallback_port_id = fallback_port_id;

    for (i = 0, entry = head; i < num_entries; i++, entry = entry->next) {
        assert(entry);

        unsigned entry_id = entry->entry_id;
        rtable_entries[entry_id].port_id = entry->port_id;

        uint32_t x;
        for (x = 0; x < (1 << IPV4_RTABLE_ENTRY_NUM_BITS); x++) {
            if (entry->children[x])
                rtable_entries[entry_id].children[x] = entry->children[x]->entry_id;
            else
                rtable_entries[entry_id].children[x] = 0;
        }
    }

    dispose_entry();
    
    return rtable;
}

uint8_t
ipv4_rtable_lookup(struct ipv4_rtable *rtable, uint32_t addr)
{
	int shift;
	unsigned entry_id = 0;
	struct ipv4_rtable_entry *rtable_entries = (struct ipv4_rtable_entry *) rtable->entries;
	uint8_t port_id = rtable->fallback_port_id;

	// Go over "addr" 4 bits at a time, starting from MSB
	for (shift = 32 - IPV4_RTABLE_ENTRY_NUM_BITS; shift >= 0; shift -= IPV4_RTABLE_ENTRY_NUM_BITS) {
		uint32_t x = (addr >> shift) & ((1 << IPV4_RTABLE_ENTRY_NUM_BITS) - 1);

		total_memory_accesses++;
		if (rtable_entries[entry_id].port_id != rtable->fallback_port_id) {
			port_id = rtable_entries[entry_id].port_id;
		}

		if (rtable_entries[entry_id].children[x]) {
			entry_id = rtable_entries[entry_id].children[x];
		} else {
			break;
		}
	}

	total_queries++;

	return port_id;
}

int batch_index;

uint8_t
ipv4_rtable_lookup_nogoto(struct ipv4_rtable *rtable, uint32_t *addr_array, uint8_t *port_id_array)
{
	foreach(batch_index, BATCH_SIZE) {
		int shift;
		unsigned entry_id = 0;
		struct ipv4_rtable_entry *rtable_entries = (struct ipv4_rtable_entry *) rtable->entries;
		uint8_t port_id = rtable->fallback_port_id;
	
		// Go over "addr" 4 bits at a time, starting from MSB
		for (shift = 32 - IPV4_RTABLE_ENTRY_NUM_BITS; shift >= 0; shift -= IPV4_RTABLE_ENTRY_NUM_BITS) {
			uint32_t x = (addr_array[batch_index] >> shift) & ((1 << IPV4_RTABLE_ENTRY_NUM_BITS) - 1);
	
			total_memory_accesses++;
			if (rtable_entries[entry_id].port_id != rtable->fallback_port_id) {
				port_id = rtable_entries[entry_id].port_id;
			}
	
			if (rtable_entries[entry_id].children[x]) {
				entry_id = rtable_entries[entry_id].children[x];
			} else {
				break;
			}
		}

		total_queries++;
		port_id_array[batch_index] = port_id;
	}

}

void
ipv4_rtable_lookup_multi(struct ipv4_rtable *rtable, uint32_t *addr_array, uint8_t *port_id_array)
{
	unsigned i;
	int shift;
	unsigned entry_id_array[16];
	uint8_t port_id_array_internal[16];
	struct ipv4_rtable_entry *rtable_entries = (struct ipv4_rtable_entry *) rtable->entries;
	char finished[16];
	uint32_t count_finished = 0;

	for (i = 0; i < BATCH_SIZE; i++) {
		port_id_array_internal[i] = rtable->fallback_port_id;
		entry_id_array[i] = 0;
		finished[i] = 0;
	}

	for (shift = 32 - IPV4_RTABLE_ENTRY_NUM_BITS; shift >= 0 && count_finished < BATCH_SIZE; shift -= IPV4_RTABLE_ENTRY_NUM_BITS) {
		for (i = 0; i < BATCH_SIZE; i++) {
			if (!finished[i]) {
				uint32_t eiai = entry_id_array[i];
				uint32_t x = (addr_array[i] >> shift) & 15;
				if (rtable_entries[eiai].port_id != rtable->fallback_port_id)
					port_id_array_internal[i] = rtable_entries[eiai].port_id;
				if (rtable_entries[eiai].children[x]) {
					eiai = rtable_entries[eiai].children[x];
					_mm_prefetch(&rtable_entries[eiai], _MM_HINT_T0);
					_mm_prefetch(((char *)&rtable_entries[eiai]) + 64, _MM_HINT_T0);
					entry_id_array[i] = eiai;
				} else {
					finished[i] = 1;
					count_finished++;
				}
			}
		}
	}

	for (i = 0; i < BATCH_SIZE; i++) {
		port_id_array[i] = port_id_array_internal[i];
	}
}

void
ipv4_rtable_print_statistics()
{
    printf("total memory accesses: %lld\n", (long long) total_memory_accesses);
    printf("total queries: %lld\n", (long long) total_queries);
    printf("average memory accesses: %.2lf\n", (double) total_memory_accesses / total_queries);
}

