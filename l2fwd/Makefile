ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

# Default target, can be overriden by command line or environment
RTE_TARGET ?= x86_64-default-linuxapp-gcc

include $(RTE_SDK)/mk/rte.vars.mk

# binary name
APP = l2fwd

# all source are stored in SRCS-y
SRCS-y := main.c common.c server.c client.c util.c

CFLAGS += -O3
CFLAGS += $(WERROR_FLAGS) -Wno-missing-prototypes -Wno-missing-declarations

include $(RTE_SDK)/mk/rte.extapp.mk
