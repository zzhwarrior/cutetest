// dma_tcm_ame_test.c — Same three-size DMA smoke test as dma_tcm_smoke.c,
// but the transfer is triggered by the AME_DMA_LOAD custom instruction
// instead of raw MMIO writes.
//
// This exercises the full path:
//   CPU issues ame_dma_load(src, dst, len)
//     -> RoCC decode routes funct=0x72 to TcmDmaCtrl inside RoCC2CUTE
//     -> TcmDmaCtrl walks its FSM: 5 MMIO PutFull32 writes to program the
//        DMA engine, one PutFull32 write of CTRL=1 to start, poll STATUS
//        via Get32 until STATUS.done, then two cleanup PutFull32 writes
//        (IRQ_CLR=1, CTRL=0).
//     -> The tile's io.cmd.ready stays low during the whole sequence, so
//        this baremetal code sees ame_dma_load block until completion.
//
// The register-level DMA behavior itself is identical to dma_tcm_smoke.c;
// only the trigger mechanism changes. So both tests share the same PASS
// criteria: byte-exact match between DRAM src[] and TCM after the copy.

#include <stdint.h>
#include <stdio.h>
#include "ame.h"

#define TCM_BASE 0x81000000ULL

static uint8_t __attribute__((aligned(64))) src[512];

static void fill_pattern(int len, uint8_t p, uint8_t q) {
    for (int i = 0; i < len; i++) src[i] = (uint8_t)(i * p + q);
}

static void poison_tcm(int len) {
    volatile uint8_t *tcm = (volatile uint8_t *)(uintptr_t)TCM_BASE;
    for (int i = 0; i < len; i++) tcm[i] = 0xAA;
    asm volatile("fence rw, rw" ::: "memory");
}

static int verify(int len) {
    volatile uint8_t *tcm = (volatile uint8_t *)(uintptr_t)TCM_BASE;
    asm volatile("fence rw, rw" ::: "memory");
    int errors = 0;
    for (int i = 0; i < len; i++) {
        uint8_t got = tcm[i];
        uint8_t exp = src[i];
        if (got != exp) {
            if (errors < 8) {
                printf("  mismatch @ %3d: expected 0x%02x, got 0x%02x\n", i, exp, got);
            }
            errors++;
        }
    }
    return errors;
}

static int one_test(const char *tag, int len, uint8_t p, uint8_t q) {
    printf("[%s] len=%d bytes  (AME_DMA_LOAD instruction)\n", tag, len);
    fill_pattern(len, p, q);
    poison_tcm(len);

    // Fence so src[] and the poison writes are globally visible before the
    // DMA reads them. Also acts as a stack-safe barrier before the RoCC
    // instruction issues.
    asm volatile("fence rw, rw" ::: "memory");

    ame_dma_load((uint64_t)(uintptr_t)src, TCM_BASE, (uint32_t)len);

    // Instruction returned means TcmDmaCtrl FSM finished the cleanup phase.
    // A fence here is defensive — the DMA writes have already retired.
    asm volatile("fence rw, rw" ::: "memory");

    int errors = verify(len);
    if (errors == 0) {
        printf("[%s] PASS\n", tag);
        return 0;
    }
    printf("[%s] FAIL: %d/%d mismatched bytes\n", tag, errors, len);
    return 1;
}

int main(void) {
    printf("=== AME_DMA_LOAD smoke test ===\n");
    printf("TCM base = 0x%08x%08x\n",
           (uint32_t)(TCM_BASE >> 32), (uint32_t)(TCM_BASE & 0xFFFFFFFFu));
    printf("src[] @ %p\n", src);

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
