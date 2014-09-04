#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "city.h"

#define IPv4_CACHE_KEY 1

#define IPv4_CACHE_CAP (512 * 1024 * 1024)		// Number of IPv4 addresses cached
#define IPv4_CACHE_CAP_ ((512 * 1024 * 1024) - 1)

void ipv4_cache_init(uint8_t **ipv4_cache);
