#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

void red_printf(const char *format, ...);
inline uint32_t fastrand(uint64_t* seed);
