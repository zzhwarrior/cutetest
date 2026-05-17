#################################
# RISCV Toolchain
#################################

CUTEROOT = ${PWD}/../..
RVVTOOLCHAIN = ${CUTEROOT}/tool/riscv
PREFIX = ${RVVTOOLCHAIN}/bin/riscv64-unknown-elf-

GCC = $(PREFIX)gcc
OBJDUMP = $(PREFIX)objdump

#################################
# Flags
#################################

ARCH = rv64imafdcv
ABI = lp64d
ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI)

CFLAGS  = -std=gnu99 -g -fno-common -fno-builtin-printf -Wall -O3
CFLAGS += $(ARCHFLAGS)
LDFLAGS = -static

libgloss_specs := htif_nano.specs
CFLAGS += -specs=$(libgloss_specs)
LDFLAGS += -specs=$(libgloss_specs)

PROGRAMS = ame_matmul_128x128x64 ame_matmul_256x256x128

.DEFAULT_GOAL := default

#################################
# Build
#################################

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

%.riscv: %.o
	$(GCC) $(LDFLAGS) $< -o $@

%.dump: %.riscv
	$(OBJDUMP) -D -S $< > $@

#################################
# Recipes
#################################

.PHONY: clean
clean:
	rm -f *.riscv *.o *.dump

.PHONY: default
default: $(addsuffix .riscv, $(PROGRAMS))

.PHONY: dumps
dumps: $(addsuffix .dump, $(PROGRAMS))
