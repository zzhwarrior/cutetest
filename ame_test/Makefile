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

PROGRAMS =ame_matmul_128x128x64 ame_matmul_256x256x128 ame_matmul_512x512x512 ame_matmul_512x512x512_dma ame_matmul_mc_512x512x512_dma mt_hello mc_tcm_rw_smoke mc_ame_store_smoke test l2_test dma_tcm_smoke dma_tcm_ame_test dma_tcm_roundtrip saturn_tcm_test tcm_pingpong_skel tcm_partition_smoke

.DEFAULT_GOAL := default

#################################
# Build
#################################

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

%.riscv: %.o
	$(GCC) $(LDFLAGS) $< -o $@

# TCM variant: places a[] and b[] directly at 0x81000000 via htif_tcm.ld,
# so no runtime memcpy is needed.
%_tcm.riscv: %.o
	$(GCC) $(LDFLAGS) -T htif_tcm.ld $< -o $@

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
