#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <assert.h>

#include "city.h"
#include "util.h"

#define IPv4_TABLE_24_KEY 1
#define IPv4_TABLE_LONG_KEY 2

#define IPv4_TABLE_LONG_CAP (64 * 1024 * 1024)		// 256 MB

struct dir_ipv4_table {
	int *tbl_24;
	int *tbl_long;
};

void dir_ipv4_init(struct dir_ipv4_table *ipv4_table, int portmask);
