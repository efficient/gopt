#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/sysinfo.h>
#include <sched.h>
#include <errno.h>
#include <xmmintrin.h>

#include "utility.h"

uint32_t 
get_num_cpus()
{
    return get_nprocs();
}

int
bind_cpu(uint32_t cpu)
{
    uint32_t n = get_num_cpus();
    int ret;

    if (cpu >= n) {
        errno = -EINVAL;
        return -1;
    }

    cpu_set_t cmask;

    CPU_ZERO(&cmask);
    CPU_SET(cpu, &cmask);

    ret = sched_setaffinity(0, sizeof(cmask), &cmask);

    return ret;
}

void 
prefetch(const void *object, uint64_t size)
{
    uint64_t offset = ((uint64_t)object) & 0x3fUL;
    const char *p = (const char *)object - offset;
    uint64_t i;
    for (i = 0; i < offset + size; i += 64)
        _mm_prefetch(p + i, _MM_HINT_T0);
}

uint64_t
time_elapsed(struct timeval *start, struct timeval *end)
{
    /* printf("start.tv_sec: %d, start.tv_usec: %d\n", start->tv_sec, start->tv_usec); */
    /* printf("end.tv_sec: %d, end.tv_usec: %d\n", end->tv_sec, end->tv_usec); */
    long long usec = (end->tv_sec - start->tv_sec) * 1000000;
    usec += (end->tv_usec - start->tv_usec);
    return (uint64_t) usec;
}
