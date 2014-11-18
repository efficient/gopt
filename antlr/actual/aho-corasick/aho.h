#include<stdint.h>
#include "ds_queue.h"

#define AHO_SHM_KEY 1

#define AHO_MAX_STATES 65536
#define AHO_FAIL 65535
#define AHO_ALPHA_SIZE 256

/**< Just some reasonable numbers */
#define AHO_MAX_PATTERNS (32 * 1024)
#define AHO_MAX_PATTERN_LEN (1024)
#define AHO_MAX_THREADS 16

#define AHO_MAX_DFA 32

struct aho_dfa {
	int id;
	int num_used_states;
	struct aho_state *root;
};

struct aho_state {
	uint16_t G[AHO_ALPHA_SIZE];		/**< Goto function */
	uint16_t F;						/**< Failure function */
	struct ds_queue output;			/**< Output patterns at this state */
	uint8_t pad[32];
};

struct aho_pattern {
	int len;
	uint8_t *content;
};

struct aho_ctrl_blk {
	int tid;						/**< Thread ID */
	struct aho_dfa **dfa_arr;		/**< The shared DFAs */
};

void aho_init(struct aho_dfa *dfa, int id);
void aho_add_pattern(struct aho_dfa *dfa, struct aho_pattern *pattern, int index);
void aho_build_ff(struct aho_dfa *dfa);
struct aho_pattern* aho_get_strings(const char *filename, int *num_patterns);
struct aho_pattern* aho_get_patterns(const char *filename, int *num_patterns);
void aho_preprocess_dfa(struct aho_dfa *dfa);
