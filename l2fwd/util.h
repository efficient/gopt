#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <assert.h>

#define LL long long
#define ULL unsigned long long

/* Some commonly used sizes */
#define K_512 524288
#define K_512_ 524287

#define M_1 1048576
#define M_1_ 1048575

#define M_2 2097152
#define M_2_ 2097151

#define M_4 4194304
#define M_4_ 4194303

#define M_8 8388608
#define M_8_ 8388607

#define M_16 16777216
#define M_16_ 16777215

#define M_32 33554432
#define M_32_ 33554431

#define M_128 134217728
#define M_128_ 134217727

#define M_256 268435456
#define M_256_ 268435455

#define M_512 536870912
#define M_512_ 536870911

#define M_1024 1073741824
#define M_1024_ 1073741823

#define M_2048 2147483648
#define M_2048_ 2147483647

#define ISSET(a, i) (a & (1 << i))
#define MAX(a, b) (a > b ? a : b)
#define htons(n) (((((unsigned short)(n) & 0xFF)) << 8) | (((unsigned short)(n) & 0xFF00) >> 8))

#define CPE2(cmp, msg, val_1, val_2) \
	if(cmp) {fflush(stdout); fprintf(stderr, msg, val_1, val_2); exit(-1);}
#define CPE1(cmp, msg, val_1) \
	if(cmp) {fflush(stdout); fprintf(stderr, msg, val_1); exit(-1);}
#define CPE(cmp, msg) \
	if(cmp) {fflush(stdout); fprintf(stderr, msg); exit(-1);}

void red_printf(const char *format, ...);
void blue_printf(const char *format, ...);
void print_buf(char *A, int n);
void *shm_alloc(int key, int bytes);
void *shm_map(int key, int bytes);
inline uint32_t fastrand(uint64_t* seed);
int bitcount(int n);
int *get_active_bits(int mask);

inline void set_mac(uint8_t *mac_ptr, ULL mac_addr);
inline ULL get_mac(uint8_t *mac_ptr);
inline void swap_mac(uint8_t *src_mac_ptr, uint8_t *dst_mac_ptr);

void print_mac_arr(uint8_t *mac);
void print_mac_ull(ULL mac);
