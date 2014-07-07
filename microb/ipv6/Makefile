TOPLEVEL = $(CURDIR)/../../

CC = gcc

# Debug
# OPT = -g -ggdb
# Release
OPT = -O3

CFLAGS += $(OPT) -march=native

LDFLAGS += $(OPT)

# DPDK
RTE_SDK = $(TOPLEVEL)/dpdk
RTE_TARGET = x86_64-default-linuxapp-$(CC)
CFLAGS += -isystem $(RTE_SDK)/$(RTE_TARGET)/include -include rte_config.h
LDFLAGS += -L$(RTE_SDK)/$(RTE_TARGET)/lib
LIBS += \
	-lrte_cmdline \
	-lrte_hash \
	-lrte_lpm \
	-lrte_malloc \
	-lrte_mbuf \
	-lrte_mempool \
	-lrte_pmd_ixgbe \
	-lrte_ring \
	-lrte_timer \
	-lrte_eal \
	-lrte_pmd_ring \
	-pthread \
	-lethdev \

LIBS += -lrt

HEADERS =

SOURCES = \
	microb.c

OBJDIR = obj

OBJS = $(patsubst %.c, $(OBJDIR)/%.o, $(SOURCES))

PROGRAMS = \
	$(OBJDIR)/microb

default: all

all: $(PROGRAMS)

clean:
	rm -rf $(PROGRAMS) $(OBJS) $(OBJDIR)/*.o

$(OBJDIR)/microb: $(OBJDIR)/microb.o $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)

$(OBJDIR)/%.o: %.c ${HEADERS} Makefile
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@
