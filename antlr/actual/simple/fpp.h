#define FPP_SET(n, i) (n | (1 << i))	// Set the ith bit of n
	
// Prefetch, Save, and Switch
#define FPP_PSS(addr, label) \
do {\
	__builtin_prefetch(addr); \
	batch_rips[I] = &&label; \
	I = (I + 1) & BATCH_SIZE_;	\
	goto *batch_rips[I]; \
} while(0)
