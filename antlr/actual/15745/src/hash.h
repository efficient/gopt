#ifndef __HASH_H_INCLUDED__
#define __HASH_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

#include <smmintrin.h>

#include "basic_types.h"
#include "utility.h"

#ifndef FALLTHROUGH_INTENDED
#define FALLTHROUGH_INTENDED do {} while (0)
#endif

static inline u32
hash_crc(const char *data, u32 n, u32 v)
{
    const char *limit = data + n;
    u32 h = 0;

    while (data + 4 <= limit) {
        u32 w = *((u32 *)data);
        data += 4;
        v = _mm_crc32_u32(w, v);
    }

    switch (limit - data) {
    case 3:
        h |= data[2] << 16;
        FALLTHROUGH_INTENDED;
    case 2:
        h |= data[1] << 8;
        FALLTHROUGH_INTENDED;
    case 1:
        h |= data[0];
        v = _mm_crc32_u32(h, v);
        break;
    }

    return v;
}

static inline u32
hash_crc_u64(u64 data, u32 v)
{
    u32 w = data & 0xFFFFFFFF;
    v = _mm_crc32_u32(w, v);
    w = data >> 32;
    v = _mm_crc32_u32(w, v);
    return v;
}

static inline u32
hash(const char *data, size_t n, u32 seed)
{
    const u32 m = 0xc6a4a793;
    const u32 r = 24;
    const char *limit = data + n;
    u32 h = seed ^ (n * m);

    while (data + 4 <= limit) {
        u32 w = *((u32 *)data);
        data += 4;
        h += w;
        h *= m;
        h ^= (h >> 16);
    }

    switch (limit - data) {
    case 3:
        h += data[2] << 16;
        FALLTHROUGH_INTENDED;
    case 2:
        h += data[1] << 8;
        FALLTHROUGH_INTENDED;
    case 1:
        h += data[0];
        h *= m;
        h ^= (h >> r);
        break;
    }
    return h;
}

u32
city_hash(const char *buf, size_t len);

static inline u32
cheap_hash(u64 i)
{
    return (i * 2654435761) & 0xFFFFFFFF;
}

#ifdef __cplusplus
}
#endif

#endif /* __HASH_H_INCLUDED_ */
