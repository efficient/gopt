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

void ipv4_cache_init(uint8_t **ipv4_cache, int portmask)
{
	int i;

	int num_active_ports = ipv4_bitcount(portmask);
	int *port_arr = ipv4_get_active_ports(portmask);

	printf("Initializing ipv4 address cache of size = %lu bytes\n", 
		IPv4_CACHE_CAP * sizeof(uint8_t));

	int sid = shmget(IPv4_CACHE_KEY, IPv4_CACHE_CAP * sizeof(uint8_t), 
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("Could not create ipv4 address cache.\n");
		exit(-1);
	}

	*ipv4_cache = shmat(sid, 0, 0);

	// Each entry in the IPv4 cache contains an enabled portid
	printf("Putting ports into ipv4 address cache randomly\n");

	for(i = 0; i < IPv4_CACHE_CAP; i++) {
		(*ipv4_cache)[i] = port_arr[rand() % num_active_ports];
	}
}
