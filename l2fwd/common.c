#include "main.h"

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

void print_mac(int port_id, struct ether_addr macaddr)
{
	printf("\tPort %u, MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n\n",
		(unsigned) port_id,
		macaddr.addr_bytes[0], macaddr.addr_bytes[1], macaddr.addr_bytes[2],
		macaddr.addr_bytes[3], macaddr.addr_bytes[4], macaddr.addr_bytes[5]);
}

void check_all_ports_link_status(uint8_t port_num, int portmask)
{
	uint8_t port_id;
	struct rte_eth_link link;

	printf("\nChecking link status\n");
	
	for (port_id = 0; port_id < port_num; port_id++) {
		if (!ISSET(portmask, port_id))
			continue;

		memset(&link, 0, sizeof(struct rte_eth_link));
		rte_eth_link_get(port_id, &link);
			
		if (link.link_status) {
			printf("Port %d Link Up \n", port_id);
		} else {
			printf("Port %d Link Down!\n", port_id);
		}
	}
}

void print_buf(char *A, int n)
{
	int i;
	for(i = 0; i < n; i++) {
		if(A[i] >= 'a' && A[i] <= 'z') {
			printf("%c, ", A[i]);
		} else {
			printf("%d, ", A[i]);
		}
	}
	printf("\n");
}

struct rte_mempool *mempool_init(char *name, int socket_id)
{
	struct rte_mempool *ret = rte_mempool_create(name, 
		NB_MBUF, MBUF_SIZE, NB_MBUF_CACHE,
		sizeof(struct rte_pktmbuf_pool_private),
		rte_pktmbuf_pool_init, NULL, rte_pktmbuf_init, NULL,
		socket_id, 0);
	
	CPE(ret == NULL, "rte_mempool_create failed\n")
	return ret;
}

int *shm_alloc(int key, int cap)
{
	int shm_flags = IPC_CREAT | 0666 | SHM_HUGETLB;
	int ht_log_sid = shmget(key, cap * sizeof(int), shm_flags);
	if(ht_log_sid == -1) {
		fprintf(stderr, "shmget Error! Failed to shm_alloc\n");
		int doh = system("cat /sys/devices/system/node/*/meminfo | grep Huge");
		exit(doh);
	}	

	int *data = (int *) shmat(ht_log_sid, 0, 0);

	int i;
	for(i = 0; i < cap; i++) {
		data[i] = rand() & LOG_CAP_;
	}
	return data;
}

inline uint32_t fastrand(uint64_t* seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}

// FAC is a cycles-to-nanoseconds conversion factor
void micro_sleep(int us, double cycles_to_ns_fac)
{
	LL required_diff = (us * 1000) / cycles_to_ns_fac;
	LL start_tsc = rte_rdtsc();
	LL end_tsc = start_tsc;

	while(end_tsc - start_tsc <= required_diff) {
		end_tsc = rte_rdtsc();
	}
}

int count_active_lcores(void)
{
	int lid, ret = 0;
	for(lid = 0; lid < RTE_MAX_LCORE; lid++) {
		if(rte_lcore_is_enabled(lid)) {
			ret ++;
		}
	}
	return ret;
}

// Get a zero-based identifier for an lcore on a socket
// Rank of an lcore is the number of active lcores with IDs stricly
// smaller than its lcore_id, on specified socket_id
int get_lcore_rank(int lcore_id, int socket_id)
{
	int rank = -1, lid;
	for(lid = 0; lid <= lcore_id; lid ++) {
		if(rte_lcore_is_enabled(lid) && 
			rte_lcore_to_socket_id(lid) == (unsigned) socket_id) {
			rank ++;
		}
	}
	return rank;	
}

int get_lcore_ranked_n(int n, int socket_id)
{
	int lid;
	for(lid = 0; lid < RTE_MAX_LCORE; lid ++) {
		if(get_lcore_rank(lid, socket_id) == n) {
			return lid;
		}
	}

	fprintf(stderr, "No lcore of rank %d on socket %d exists. Exiting.\n",
		n, socket_id);
	exit(-1);
}

// Count the number of 1-bits in n
int bitcount(int n)
{
	int count = 0;
	while(n > 0) {
		count ++;
		n = n & (n - 1);
	}
	return count;
}

// Returns an array containing the port numbers of all ports that are active
int *get_active_ports(int portmask)
{
	int num_active_ports = bitcount(portmask);
	int *active_ports = (int *) malloc(num_active_ports * sizeof(int));
	int pos = 0, i;
	for(i = 0; i < RTE_MAX_ETHPORTS; i++) {
		if(ISSET(portmask, i)) {
			active_ports[pos] = i;
			pos ++;
		}
	}
	assert(pos == num_active_ports);
	return active_ports;
}

int count_active_lcores_on_socket(int socket_id)
{
	int active_lcores = 0;
	int i = 0;
	for(i = 0; i < RTE_MAX_LCORE; i++) {
		if(rte_lcore_is_enabled(i) && rte_lcore_to_socket_id(i) == (unsigned) socket_id) {
			active_lcores ++;
		}
	}
	return active_lcores;
}

// On CPUs that use an I/O Hub, rte_eth_dev_socket_id(port_id) returns -1.
// So, we assign a socket manually using macaddr. XXX: XIA-specific
int get_socket_id_from_macaddr(int port_id)
{
	struct ether_addr macaddr;
	rte_eth_macaddr_get(port_id, &macaddr);

	int lsb = macaddr.addr_bytes[5];
	
	// Least byte of mac addr of xge0 and xge1 on xia-router1 and xia-router0
	if(lsb == 0x36 || lsb == 0x37 || lsb == 0x44 || lsb == 0x45) {
		return 0;
	}

	// Least byte of mac addr of xge2 and xge3 on xia-router1 and xia-router0
	if(lsb == 0xA8 || lsb == 0xA9 || lsb == 0x0A || lsb == 0x0B) {
		return 1;
	}

	red_printf("Unexpected mac addr ending: %x\n", lsb);
	exit(-1);
}

// Return the client lcore responsible for queue #queue_id on port #port_id
// XXX: XIA-specific
int client_port_queue_to_lcore(int port_id, int queue_id)
{
	assert(port_id <= 3 && port_id >= 0);
	assert(queue_id <= 2 && queue_id >=0);

	int mapping[4][3] = {{0, 2, 4},		// xge0: lcores 0, 2, 4
						 {6, 8, 10},		// xge1: lcores 6, 8, 10
						 {1, 3, 5},		// xge2: lcores 1, 3, 5
						 {7, 9, 11}};	// xge3: lcores 7, 9, 11

	return mapping[port_id][queue_id];
}

void set_mac(uint8_t *mac_ptr, LL mac_addr)
{
   	mac_ptr[0] = mac_addr & 0xFF;
    mac_ptr[1] = (mac_addr >> 8) & 0xFF;
    mac_ptr[2] = (mac_addr >> 16) & 0xFF;
    mac_ptr[3] = (mac_addr >> 24) & 0xFF;
    mac_ptr[4] = (mac_addr >> 32) & 0xFF;
    mac_ptr[5] = (mac_addr >> 40) & 0xFF;
}

void swap_mac(uint8_t *src_mac_ptr, uint8_t *dst_mac_ptr)
{
	int i = 0;
	for(i = 0; i < 6; i ++) {
		uint8_t temp = src_mac_ptr[i];
		src_mac_ptr[i] = dst_mac_ptr[i];
		dst_mac_ptr[i] = temp;
	}
}

void print_ether_hdr(struct ether_hdr *eth_hdr)
{
	int i;
	printf("Ether hdr:\n");
	printf("\tDst mac: ");
	for(i = 0; i < 6; i ++) {
		printf("%x ", eth_hdr->d_addr.addr_bytes[i]);
	}
	printf("\n\tSrc mac: ");
	for(i = 0; i < 6; i ++) {
		printf("%x ", eth_hdr->s_addr.addr_bytes[i]);
	}
	printf("\n");
}

