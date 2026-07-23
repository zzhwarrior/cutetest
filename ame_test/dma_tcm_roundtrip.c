// dma_tcm_roundtrip.c — validate DMA in BOTH directions:
//   Phase A: DRAM src[] --(ame_dma_load)--> TCM
//   Phase B: TCM       --(ame_dma_load)--> DRAM dst[]
//   Verify DRAM dst[] == DRAM src[] byte-for-byte.
//
// This checks that TcmDmaEngine works TCM→memory correctly (the engine
// itself is direction-agnostic; the CTRL[2]=dir bit reserved in the plan
// isn't even needed — putting TCM address in SRC and DRAM address in DST
// simply routes reads to tcmNode and writes to cacheNode).

#include <stdint.h>
#include <stdio.h>
#include "ame.h"

#define TCM_BASE 0x81000000ULL

static uint8_t __attribute__((aligned(64))) src[512];
static uint8_t __attribute__((aligned(64))) dst[512];

static void fill_pattern(int len, uint8_t p, uint8_t q) {
    for (int i = 0; i < len; i++) src[i] = (uint8_t)(i * p + q);
    for (int i = 0; i < len; i++) dst[i] = 0x00;
}

static int compare(int len) {
    int errors = 0;
    for (int i = 0; i < len; i++) {
        if (dst[i] != src[i]) {
            if (errors < 8) {
                printf("  mismatch @ %3d: src=0x%02x dst=0x%02x\n",
                       i, src[i], dst[i]);
            }
            errors++;
        }
    }
    return errors;
}

static void poison_tcm(int len) {
    volatile uint8_t *tcm = (volatile uint8_t *)(uintptr_t)TCM_BASE;
    for (int i = 0; i < len; i++) tcm[i] = 0xEE;
    asm volatile("fence rw, rw" ::: "memory");
}

static int one_test(const char *tag, int len, uint8_t p, uint8_t q) {
    printf("[%s] len=%d bytes\n", tag, len);
    fill_pattern(len, p, q);
    poison_tcm(len);

    asm volatile("fence rw, rw" ::: "memory");

    // Phase A: DRAM src -> TCM
    ame_dma_load((uint64_t)(uintptr_t)src, TCM_BASE, (uint32_t)len);
    asm volatile("fence rw, rw" ::: "memory");

    // Phase B: TCM -> DRAM dst   (this exercises the reverse direction)
    ame_dma_load(TCM_BASE, (uint64_t)(uintptr_t)dst, (uint32_t)len);
    asm volatile("fence rw, rw" ::: "memory");

    int errors = compare(len);
    if (errors == 0) {
        printf("[%s] PASS\n", tag);
        return 0;
    }
    printf("[%s] FAIL: %d/%d bytes mismatched\n", tag, errors, len);
    return 1;
}

int main(void) {
    printf("=== TCM DMA roundtrip test ===\n");
    printf("Phase A: DRAM src[] -> TCM\n");
    printf("Phase B: TCM        -> DRAM dst[]\n");
    printf("src[] @ %p\n", src);
    printf("dst[] @ %p\n", dst);

    int fail = 0;
    fail |= one_test("test1_64B",   64,  0x11, 0x55);
    fail |= one_test("test2_128B", 128,  0x07, 0x03);
    fail |= one_test("test3_512B", 512,  0x1D, 0x2A);

    if (fail) {
        printf("\n=== OVERALL: FAIL ===\n");
        return 1;
    }
    printf("\n=== OVERALL: PASS ===\n");
    return 0;
}
