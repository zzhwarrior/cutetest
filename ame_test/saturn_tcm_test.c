// saturn_tcm_test.c — verify Saturn vector loads/stores reach HuanCun TCM.
//
// Path exercised: Saturn atlNode → tcmAdjuster* → TLBuffer → tlMasterXbar →
// tcmNode → TcmSinkA → DataStorage TCM SRAM (and back on d channel).
//
// The vector unit accesses TCM as an ordinary UNCACHED memory region; there
// is no special instruction needed — just point vle/vse at 0x81000000+.

#include <stdint.h>
#include <stdio.h>
#include <riscv_vector.h>
#include "ame.h"

#define TCM_BASE 0x81000000ULL

static uint32_t __attribute__((aligned(64))) src[128];
static uint32_t __attribute__((aligned(64))) dst[128];

static void fill(int n, uint32_t p) {
    for (int i = 0; i < n; i++) src[i] = (i * p) ^ 0xA5A5A5A5u;
    for (int i = 0; i < n; i++) dst[i] = 0;
}

// vector copy of `n` uint32_t from `s` to `d` using rvv 1.0 intrinsics.
static void vcopy_u32(uint32_t *d, const uint32_t *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vuint32m1_t v = __riscv_vle32_v_u32m1(s + i, vl);
        __riscv_vse32_v_u32m1(d + i, v, vl);
        i += vl;
    }
}

int main(void) {
    printf("=== Saturn vector -> HuanCun TCM test ===\n");
    volatile uint32_t *tcm = (volatile uint32_t *)(uintptr_t)TCM_BASE;
    const int N = 128;

    fill(N, 0x1D);

    // 1) Vector store: DRAM src[] -> TCM via Saturn (not via CPU scalar stores)
    asm volatile ("fence rw, rw" ::: "memory");
    vcopy_u32((uint32_t *)TCM_BASE, src, N);
    asm volatile ("fence rw, rw" ::: "memory");

    // 2) Scalar readback: does TCM hold the same words we wrote?
    int errs1 = 0;
    for (int i = 0; i < N; i++) {
        if (tcm[i] != src[i]) {
            if (errs1 < 4) printf("  wr@%d: exp 0x%08x got 0x%08x\n", i, src[i], tcm[i]);
            errs1++;
        }
    }
    printf("Phase A (vector store to TCM): %s (%d errors)\n",
           errs1 == 0 ? "PASS" : "FAIL", errs1);

    // 3) Vector load: TCM -> DRAM dst[]
    vcopy_u32(dst, (const uint32_t *)TCM_BASE, N);
    asm volatile ("fence rw, rw" ::: "memory");

    int errs2 = 0;
    for (int i = 0; i < N; i++) {
        if (dst[i] != src[i]) {
            if (errs2 < 4) printf("  rd@%d: exp 0x%08x got 0x%08x\n", i, src[i], dst[i]);
            errs2++;
        }
    }
    printf("Phase B (vector load  from TCM): %s (%d errors)\n",
           errs2 == 0 ? "PASS" : "FAIL", errs2);

    // 4) End-to-end via DMA: DRAM->TCM (DMA), TCM->DRAM (vector load).
    for (int i = 0; i < N; i++) dst[i] = 0;
    ame_dma_load((uint64_t)(uintptr_t)src, TCM_BASE, N * sizeof(uint32_t));
    asm volatile ("fence rw, rw" ::: "memory");
    vcopy_u32(dst, (const uint32_t *)TCM_BASE, N);
    asm volatile ("fence rw, rw" ::: "memory");

    int errs3 = 0;
    for (int i = 0; i < N; i++) {
        if (dst[i] != src[i]) {
            if (errs3 < 4) printf("  mix@%d: exp 0x%08x got 0x%08x\n", i, src[i], dst[i]);
            errs3++;
        }
    }
    printf("Phase C (DMA store, vector load): %s (%d errors)\n",
           errs3 == 0 ? "PASS" : "FAIL", errs3);

    int total = errs1 + errs2 + errs3;
    printf("\n=== OVERALL: %s (%d total errors) ===\n",
           total == 0 ? "PASS" : "FAIL", total);
    return total == 0 ? 0 : 1;
}
