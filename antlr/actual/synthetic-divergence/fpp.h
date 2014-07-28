#define PREFETCH(x)	{}					// Just a hint
#define FPP_SET(n, i) (n | (1 << i))	// Set the ith bit of n
#define FPP_ISSET(n, i) (n & (1 << i))
	
// Prefetch, Save, and Switch
#define FPP_PSS(addr, label) \
do {\
	__builtin_prefetch(addr); \
	batch_rips[I] = &&label; \
	I = (I + 1) & BATCH_SIZE_;	\
	goto *batch_rips[I]; \
} while(0)
