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

uint64_t
time_elapsed(struct timeval *start, struct timeval *end)
{
    /* printf("start.tv_sec: %d, start.tv_usec: %d\n", start->tv_sec, start->tv_usec); */
    /* printf("end.tv_sec: %d, end.tv_usec: %d\n", end->tv_sec, end->tv_usec); */
    long long usec = (end->tv_sec - start->tv_sec) * 1000000;
    usec += (end->tv_usec - start->tv_usec);
    return (uint64_t) usec;
}
