#################################
# RISCV Toolchain
#################################

CUTEROOT = ${PWD}/../..
RVVTOOLCHAIN = ${CUTEROOT}/tool/riscv
PREFIX = ${RVVTOOLCHAIN}/bin/riscv64-unknown-elf-

GCC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size

sysroot = $(shell $(GCC) -print-sysroot)
# GCC = clang
# sysroot = /opt/rv64gcv_lp64d_enable_llvm/riscv64-unknown-elf
#################################
# Flags
#################################

# SoC Settings
ARCH = rv64imafdcv
ABI = lp64d
ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI)

CFLAGS  = -std=gnu99 -g -fno-common -fno-builtin-printf -fpermissive -Wall -O3
CFLAGS += $(ARCHFLAGS)
LDFLAGS = -static

# include libgloss.mk

libgloss_specs := htif_nano.specs
CFLAGS += -specs=$(libgloss_specs)
LDFLAGS += -specs=$(libgloss_specs)


# PROGRAMS = pwm blkdev accum charcount nic-loopback big-blkdev pingd \
        #    streaming-passthrough streaming-fir nvdla spiflashread spiflashwrite fft gcd \
        #    hello mt-hello 8
# PROGRAMS = conv_MMA_padding_scalar_10  conv_MMA_padding_scalar_23 conv_MMA_padding_scalar_11  conv_MMA_padding_scalar_3 conv_MMA_padding_scalar_12  conv_MMA_padding_scalar_4 conv_MMA_padding_scalar_13  conv_MMA_padding_scalar_5 conv_MMA_padding_scalar_14  conv_MMA_padding_scalar_6 conv_MMA_padding_scalar_15  conv_MMA_padding_scalar_7 conv_MMA_padding_scalar_16  conv_MMA_padding_scalar_8 conv_MMA_padding_scalar_17  conv_MMA_padding_scalar_9 conv_MMA_padding_scalar_18  conv_MMA_padding_scalar_19  conv_MMA_padding_scalar_2   conv_MMA_padding_scalar_20  conv_MMA_padding_scalar_21  conv_MMA_padding_scalar_1     conv_MMA_padding_scalar_22
PROGRAMS = cutehello cute_Matmul_mnk_128_128_128_zeroinit cute_conv_mnk_49_128_128_k1_s1_oh7

.DEFAULT_GOAL := default


#################################
# Build
#################################


%.o: %.S
	$(GCC) $(CFLAGS) -D__ASSEMBLY__=1 -c $< -o $@

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

%.riscv: %.o $(libgloss)
	$(GCC) $(LDFLAGS) $< -o $@

%.dump: %.riscv
	$(OBJDUMP) -D -S $< > $@

# gcc get_test.c -o get_value
get_matrix_value: get_matrix_test.c
	$(CC) -o $@ $^

get_conv_value: get_conv_test.c
	$(CC) -o $@ $^

#################################
# Recipes
#################################

.PHONY: clean
clean:
	rm -f *.riscv *.o *.dump *.S
	$(if $(libgloss),rm -rf $(libgloss_builddir)/)

.PHONY: default
default: $(addsuffix .riscv, $(PROGRAMS))

.PHONY: dumps
dumps: $(addsuffix .dump, $(PROGRAMS))
