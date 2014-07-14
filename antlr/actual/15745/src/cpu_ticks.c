#include <time.h>
#include <sys/time.h>

#include "cpu_ticks.h"

static int
get_cpu_ticks_freq_from_clock(uint64_t *cpu_ticks_freq)
{
#ifdef CLOCK_MONOTONIC_RAW
#define NS_PER_SEC (1e9)

    struct timespec sleeptime = {.tv_nsec = 5e8};
    struct timespec t_start, t_end;

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t_start) == 0) {
        uint64_t ns, end, start = read_cpu_ticks();
        nanosleep(&sleeptime, NULL);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);
        end = read_cpu_ticks();
        ns = (t_end.tv_sec - t_start.tv_sec) * NS_PER_SEC + 
            (t_end.tv_nsec - t_start.tv_nsec);
        *cpu_ticks_freq = (uint64_t)(end - start) / ((double)ns / NS_PER_SEC);
        return 0;
    }
#endif

    return -1;
}

static void
get_cpu_ticks_freq_fallback(uint64_t *cpu_ticks_freq)
{
    uint64_t start = read_cpu_ticks();
    sleep(1);
    *cpu_ticks_freq = read_cpu_ticks() - start;
}

uint64_t
get_cpu_ticks_freq()
{
    uint64_t cpu_ticks_freq;

    if (get_cpu_ticks_freq_from_clock(&cpu_ticks_freq) < 0)
        get_cpu_ticks_freq_fallback(&cpu_ticks_freq);

    return cpu_ticks_freq;
}

