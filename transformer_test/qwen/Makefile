#################################
# RISCV Toolchain
#################################

CUTEROOT = ${PWD}/../../..
RVVTOOLCHAIN = ${CUTEROOT}/tool/riscv
PREFIX = ${RVVTOOLCHAIN}/bin/riscv64-unknown-elf-

GCC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size

sysroot = $(shell $(GCC) -print-sysroot)

#################################
# Flags
#################################

# SoC Settings
ARCH = rv64imafdcv_zvfh_zfh
ABI = lp64d
ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI)

CFLAGS  = -std=gnu99 -g -fno-common -fno-builtin-printf -fpermissive -Wall -O3
CFLAGS += $(ARCHFLAGS)
LDFLAGS = -static

libgloss_specs := htif_nano.specs
CFLAGS += -specs=$(libgloss_specs)
LDFLAGS += -specs=$(libgloss_specs)

PROGRAMS = nvfp4_smoke ame_int8_smoke ame_nvfp4_smoke_phaseA ame_nvfp4_smoke_phaseB ame_nvfp4_smoke_phaseD ame_nvfp4_smoke_phaseD_v1 ame_nvfp4_smoke_phaseD_v2 ame_nvfp4_smoke_phaseD_v3 ame_nvfp4_gemm_e256 ame_nvfp4_gemm_e256_full ame_nvfp4_gemm_e512 ame_bf16_smoke ame_bf16_gemm_e256_full ame_nvfp4_gemm_kt ame_nvfp4_wrapper_test ame_nvfp4_wrapper_g2_test ame_bf16_wrapper_test ame_bf16_wrapper_g4_test qwen_full_attn_nvfp4 qwen_full_attn_nvfp4_ame

.DEFAULT_GOAL := default

#################################
# Build
#################################

%.o: %.S
	$(GCC) $(CFLAGS) -D__ASSEMBLY__=1 -c $< -o $@

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

# Variant builds for phase D (share ame_nvfp4_smoke_phaseD.c, different -DPHASED_VARIANT)
ame_nvfp4_smoke_phaseD_v1.o: ame_nvfp4_smoke_phaseD.c
	$(GCC) $(CFLAGS) -DPHASED_VARIANT=1 -c $< -o $@
ame_nvfp4_smoke_phaseD_v2.o: ame_nvfp4_smoke_phaseD.c
	$(GCC) $(CFLAGS) -DPHASED_VARIANT=2 -c $< -o $@
ame_nvfp4_smoke_phaseD_v3.o: ame_nvfp4_smoke_phaseD.c
	$(GCC) $(CFLAGS) -DPHASED_VARIANT=3 -c $< -o $@

# Step-sliced qwen_full_attn variants (share source, differ in -DRUN_UPTO_STEP=N).
# Usage: make qwen_full_attn_nvfp4_ame_s2.riscv   (runs steps 1..2 only)
qwen_full_attn_nvfp4_ame_s%.o: qwen_full_attn_nvfp4_ame.c
	$(GCC) $(CFLAGS) -DRUN_UPTO_STEP=$* -c $< -o $@

%.riscv: %.o
	$(GCC) $(LDFLAGS) $< -o $@

%.dump: %.riscv
	$(OBJDUMP) -D -S $< > $@

#################################
# Recipes
#################################

.PHONY: clean
clean:
	rm -f *.riscv *.o *.dump *.S

.PHONY: default
default: $(addsuffix .riscv, $(PROGRAMS))

.PHONY: dumps
dumps: $(addsuffix .dump, $(PROGRAMS))
