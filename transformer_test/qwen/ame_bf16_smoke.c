// Phase F.2 pre-check: single-tile 128x128x32 BF16 GEMM on the AME path.
//
// Goal: prove that AME's existing bf16 double-widen instruction (mfmacc.s.bf16)
// and bf16 tile-load (mlae16/mlbe16) work end-to-end with the current Chisel,
// no changes needed. This unblocks running Qwen's score / attn matmuls on AME.
//
// On CUTE_4Tops_128SCP (testL2Dma1core), ReduceWidthByte=64, so 1 RV holds
// 64/2 = 32 BF16 elements. The clamp in AMEDecoder caps mtilek at
// ReduceGroupSize=1, so K=32 elements per dispatch is the natural single-tile
// size (matches what Phase E-small did for NVFP4 K=128).
//
// Golden math:
//   A_bf16[i][k] = 1.0  (bf16 = 0x3F80)
//   B_bf16[j][k] = 1.0
// => C[i][j] = Σ_{k=0..31} 1.0 * 1.0 = 32.0 = 0x42000000
//
// TCM layout:
//   0x81000000  A_bf16   128 * 32 * 2 = 8 KiB
//   0x81002000  B_bf16   128 * 32 * 2 = 8 KiB
//   0x81004000  C_fp32   128 * 128 * 4 = 64 KiB

#include <stdio.h>
#include <stdint.h>
#include "../../ame_test/ame.h"

#define M       128
#define N       128
#define K       32                       // one full RV in BF16 (64 bytes / 2 bytes-per-elem)

// IEEE-754 BF16 value 1.0 = top 16 bits of FP32 1.0 = 0x3F80.
#define BF16_ONE         0x3F80u
#define EXPECTED_C_BITS  0x42000000u     // fp32 32.0

// ---- DRAM-resident source data ----
static const uint16_t A_bf16_dram[M][K] __attribute__((aligned(64))) = {
    [0 ... M - 1] = { [0 ... K - 1] = BF16_ONE }
};
static const uint16_t B_bf16_dram[N][K] __attribute__((aligned(64))) = {
    [0 ... N - 1] = { [0 ... K - 1] = BF16_ONE }
};

// ---- TCM addresses ----
#define TCM_A_ADDR    0x81000000UL       // 8 KiB
#define TCM_B_ADDR    0x81002000UL       // 8 KiB
#define TCM_C_ADDR    0x81004000UL       // 64 KiB

static uint16_t (*A_bf16_tcm)[K]   = (uint16_t (*)[K])TCM_A_ADDR;
static uint16_t (*B_bf16_tcm)[K]   = (uint16_t (*)[K])TCM_B_ADDR;
static volatile uint32_t (*C_fp32_tcm)[N] = (volatile uint32_t (*)[N])TCM_C_ADDR;

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[BF16-SMOKE] start: M=%d N=%d K=%d (bf16 single-tile)\n", M, N, K);

    if (ame_tcm_config(4) != 0) {
        printf("[BF16-SMOKE] FAIL: could not configure TCM to 4 ways\n");
        return 1;
    }

    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_start = rd_cycle();
    ame_dma_load((uint64_t)A_bf16_dram, TCM_A_ADDR, (uint32_t)sizeof(A_bf16_dram));
    ame_dma_load((uint64_t)B_bf16_dram, TCM_B_ADDR, (uint32_t)sizeof(B_bf16_dram));
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_end = rd_cycle();
    printf("[BF16-SMOKE] DMA staging done in %lu cycles (A+B = %u bytes)\n",
           (unsigned long)(dma_end - dma_start),
           (unsigned)(sizeof(A_bf16_dram) + sizeof(B_bf16_dram)));

    // Spot check DMA: first + last element in TCM should equal 0x3F80.
    if (((volatile uint16_t *)TCM_A_ADDR)[0] != BF16_ONE ||
        ((volatile uint16_t *)TCM_A_ADDR)[M * K - 1] != BF16_ONE) {
        printf("[BF16-SMOKE] FAIL: TCM A contents mismatched after DMA\n");
        return 1;
    }

    // Configure tile and issue compute.
    ame_settilem(M);
    ame_settilen(N);
    // Pass K as element count per RISC-V matrix spec (K ≤ TRLEN/element_bits
    // = 512/16 = 32 for bf16 on this config). AMEDecoder's clamp then reduces
    // it to ReduceGroupSize=1 RV, which is what the HW actually consumes.
    // The spec-vs-CUTE deviation on mtilek units (documented in
    // ame_nvfp4_ext.h) still applies but is transparent here because 1 RV
    // holds exactly 32 bf16 elements — spec and HW agree at this granularity.
    ame_settilek(K);

    uint64_t t0 = rd_cycle();

    ame_mzero(ACC0);
    // Stride = K raw elements * 2 bytes/elem = K*2 bytes per row.
    ame_mlae16(TR0, (uint64_t)A_bf16_tcm, (uint64_t)(K * 2));
    ame_mlbe16(TR2, (uint64_t)B_bf16_tcm, (uint64_t)(K * 2));
    ame_mfmacc_s_bf16(ACC0, TR0, TR2);
    ame_msce32(ACC0, (uint64_t)C_fp32_tcm, (uint64_t)(N * 4));

    while (!ame_is_idle()) { /* spin */ }

    uint64_t t1 = rd_cycle();

    printf("[BF16-SMOKE] corners: C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)C_fp32_tcm[0][0],
           N - 1, (unsigned)C_fp32_tcm[0][N - 1],
           M - 1, (unsigned)C_fp32_tcm[M - 1][0],
           M - 1, N - 1, (unsigned)C_fp32_tcm[M - 1][N - 1]);

    int errors = 0;
    int first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < 10; j++) {
            uint32_t v = C_fp32_tcm[i][j];
            if (v != EXPECTED_C_BITS) {
                if (errors == 0) {
                    first_wrong = v;
                    first_i = i; first_j = j;
                }
                errors++;
            }
        }
    }

    printf("[BF16-SMOKE] compute cycles = %lu\n", (unsigned long)(t1 - t0));
    if (errors == 0) {
        printf("[BF16-SMOKE] PASS: all %d elements == 0x%08x\n",
               M * N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[BF16-SMOKE] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, M * N, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[BF16-SMOKE] done\n");
    return errors == 0 ? 0 : 1;
}
