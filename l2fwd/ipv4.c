#include "ipv4.h"

// Don't want to include rte header
#define IPv4_MAX_ETHPORTS 16
#define IPv4_ISSET(a, i) (a & (1 << i))

int ipv4_bitcount(int n);
int *ipv4_get_active_ports(int portmask);

// Count the number of 1-bits in n
int ipv4_bitcount(int n)
{
	int count = 0;
	while(n > 0) {
		count ++;
		n = n & (n - 1);
	}
	return count;
}

// Returns an array containing the port numbers of all ports that are active
int *ipv4_get_active_ports(int portmask)
{
	int num_active_ports = ipv4_bitcount(portmask);
	int *active_ports = (int *) malloc(num_active_ports * sizeof(int));
	int pos = 0, i;
	for(i = 0; i < IPv4_MAX_ETHPORTS; i++) {
		if(IPv4_ISSET(portmask, i)) {
			active_ports[pos] = i;
			pos ++;
		}
	}
	assert(pos == num_active_ports);
	return active_ports;
}

void dir_ipv4_init(struct dir_ipv4_table *ipv4_table, int portmask)
{
	int i;

	int tbl_24_bytes = (1 << 24) * sizeof(int);
	int tbl_long_bytes = IPv4_TABLE_LONG_CAP * sizeof(int);

	int num_active_ports = ipv4_bitcount(portmask);
	int *port_arr = ipv4_get_active_ports(portmask);

	printf("Initializing DIR-24-8-BASIC lookup table\n"
		"\ttbl_24: %d bytes, tbl_long: %d bytes\n",
		tbl_24_bytes, tbl_long_bytes);

	ipv4_table->tbl_24 = shm_alloc(IPv4_TABLE_24_KEY, tbl_24_bytes);
	ipv4_table->tbl_long = shm_alloc(IPv4_TABLE_LONG_KEY, tbl_long_bytes);

	/** < Fill both tables with random ports */
	for(i = 0; i < (1 << 24); i ++) {
		ipv4_table->tbl_24[i] = port_arr[rand() % num_active_ports];
	}

	for(i = 0; i < IPv4_TABLE_LONG_CAP; i ++) {
		ipv4_table->tbl_long[i] = port_arr[rand() % num_active_ports];
	}

	/** < Fill 3% of the in the direct table with pointers */
	for(i = 0; i < (3 * (1 << 24)) / 100; i ++) {
		int invalidate_idx = rand() % (1 << 24);
		ipv4_table->tbl_24[invalidate_idx] = (0x8000) | 
			rand() % IPv4_TABLE_LONG_CAP;
	}
}
