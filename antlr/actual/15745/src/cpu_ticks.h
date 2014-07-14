#include<stdint.h>

static inline uint64_t
read_cpu_ticks(void)
{
    uint32_t lo, hi;
    __asm__ __volatile("rdtsc"
                       : "=a" (lo),
                         "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

