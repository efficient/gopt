#include <stdint.h>

#define IPV4_RTABLE_ENTRY_NUM_BITS (4)

#define IPV4_RTABLE_SID 1		// shmid

struct ipv4_rib_entry {
    uint32_t addr;
    uint32_t netmask_num_bits;
    uint32_t netmask;
    uint8_t port_id;
};

struct ipv4_rtable {
    uint32_t n;
    uint8_t fallback_port_id;
    struct ipv4_rtable_entry * entries;
};

struct ipv4_rtable_entry {
    uint8_t port_id;
    uint32_t children[16];
};

struct ipv4_rtable *
ipv4_rtable_create(struct ipv4_rib_entry *rib_entries, unsigned n, uint8_t fallback_port_id);

uint8_t
ipv4_rtable_lookup(struct ipv4_rtable *rtable, uint32_t addr);

void
ivp4_rtable_lookup_multi(struct ipv4_rtable *table, uint32_t *addr_array, uint8_t *port_id_array);

void
ipv4_rtable_print_statistics();
