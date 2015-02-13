/* Utility functions, independent of DPDK */
#include "util.h"

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

// Like printf, but red. Limited to 1000 characters.
void blue_printf(const char *format, ...)
{	
	#define BLUE_LIM 1000
	va_list args;
	int i;

	char buf1[BLUE_LIM], buf2[BLUE_LIM];
	memset(buf1, 0, BLUE_LIM);
	memset(buf2, 0, BLUE_LIM);

    va_start(args, format);

	// Marshal the stuff to print in a buffer
	vsnprintf(buf1, BLUE_LIM, format, args);

	// Probably a bad check for buffer overflow
	for(i = BLUE_LIM - 1; i >= BLUE_LIM - 50; i --) {
		assert(buf1[i] == 0);
	}

	// Add markers for red color and reset color
	snprintf(buf2, 1000, "\033[34m%s\033[0m", buf1);

	// Probably another bad check for buffer overflow
	for(i = BLUE_LIM - 1; i >= BLUE_LIM - 50; i --) {
		assert(buf2[i] == 0);
	}

	printf("%s", buf2);

    va_end(args);
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

void *shm_alloc(int key, int bytes)
{
	int shm_flags = IPC_CREAT | 0666 | SHM_HUGETLB;
	int sid = shmget(key, bytes, shm_flags);
	if(sid == -1) {
		fprintf(stderr, "shmget Error! Failed to shm_alloc.\n");
		int doh = system("cat /sys/devices/system/node/*/meminfo | grep Huge");
		exit(doh);
	}	

	void *data = shmat(sid, 0, 0);
	assert(data != NULL);

	memset((char *) data, 0, bytes);

	return data;
}

void *shm_map(int key, int bytes)
{
	int sid = shmget(key, M_2, SHM_HUGETLB | 0666);
	if(sid == -1) {
		fprintf(stderr, "shmget Error! Failed to shm_map\n");
		int doh = system("cat /sys/devices/system/node/*/meminfo | grep Huge");
		exit(doh);
	}	
	
	void *data = shmat(sid, 0, 0);
	assert(data != NULL);

	memset((char *) data, 0, bytes);
	
	return data;
}

inline uint32_t fastrand(uint64_t* seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t) (*seed >> 32);
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

// Returns an array containing the indexes of active bits. 
// LSB's index is 0.
int *get_active_bits(int mask)
{
	int num_active_bits = bitcount(mask);
	int *active_bits = (int *) malloc(num_active_bits * sizeof(int));

	int pos = 0, i;
	for(i = 0; i < 31; i++) {			// Check all (int) bits
		if(ISSET(mask, i)) {
			active_bits[pos] = i;
			pos ++;
		}
	}

	assert(pos == num_active_bits);

	return active_bits;
}

inline void set_mac(uint8_t *mac_ptr, ULL mac_addr)
{
   	mac_ptr[0] = mac_addr & 0xFF;
    mac_ptr[1] = (mac_addr >> 8) & 0xFF;
    mac_ptr[2] = (mac_addr >> 16) & 0xFF;
    mac_ptr[3] = (mac_addr >> 24) & 0xFF;
    mac_ptr[4] = (mac_addr >> 32) & 0xFF;
    mac_ptr[5] = (mac_addr >> 40) & 0xFF;
}

inline ULL get_mac(uint8_t *mac_ptr)
{
	ULL ret = 0;
	ret = mac_ptr[0] + 
		((ULL) mac_ptr[1] << 8) + 
		((ULL) mac_ptr[2] << 16) +
		((ULL) mac_ptr[3] << 24) +
		((ULL) mac_ptr[4] << 32) +
		((ULL) mac_ptr[5] << 40);

	return ret;
}

inline void swap_mac(uint8_t *src_mac_ptr, uint8_t *dst_mac_ptr)
{
	int i = 0;
	for(i = 0; i < 6; i ++) {
		uint8_t temp = src_mac_ptr[i];
		src_mac_ptr[i] = dst_mac_ptr[i];
		dst_mac_ptr[i] = temp;
	}
}

void print_mac_arr(uint8_t *mac)
{
	printf("%02X:%02X:%02X:%02X:%02X:%02X\n",
		mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void print_mac_ull(ULL mac)
{
	uint8_t __mac[6];
	set_mac(__mac, mac);
	print_mac_arr(__mac);
}

