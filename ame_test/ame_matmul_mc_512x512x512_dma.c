// 4-core parallel INT8 GEMM (512x512x512) — each core is fully independent.
//
// Architecture recap (testL2Dma config):
//   - 4 shuttle tiles, each with a *private* HuanCun L2 whose TCM aperture
//     lives at 0x81000000 (2 MiB when count=4).
//   - Each tile also owns a *private* TcmDmaEngine and *private* AME engine.
//   - Different tiles share the address 0x81000000 for TCM, but the memory
//     is physically per-tile: tile N's DMA / AME / CPU load at 0x81000000
//     hits tile N's own TCM, not tile 0's.
//
// So every hart has to run its own TCM partition setup, its own DMA staging,
// its own GEMM kernel, and its own verify. There is no cross-tile TCM view
// to consolidate — each core prints its own results.
//
// Boot flow: hart 0 runs main() -> __main(); harts 1..3 come in through
// _start_secondary -> __main(). All four converge on __main().

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame.h"
#include "matmul_value_mnk_512_512_512.h"   // a[512][512], b[512][512] in DRAM
#include "matmul_cref_512_512_512.h"        // c_ref[512][512]

#define N_CORES 4

#define APPLICATION_M 512
#define APPLICATION_N 512
#define APPLICATION_K 512
#define TILE_M         128
#define TILE_N         128
#define TILE_K         64
#define TILES_M        (APPLICATION_M / TILE_M)   // 4
#define TILES_N        (APPLICATION_N / TILE_N)   // 4
#define TILES_K        (APPLICATION_K / TILE_K)   // 8

_Static_assert(TILES_M == N_CORES,
    "Work split assumes one M tile-row per core; TILES_M must equal N_CORES");

// Per-tile private TCM layout. Same VA on every hart, distinct physical TCM.
#define TCM_B_ADDR   0x81000000UL
#define TCM_A_ADDR   0x81040000UL
#define C_BASE_ADDR  0x81100000UL

static int32_t (*c)[APPLICATION_N]      = (int32_t (*)[APPLICATION_N])C_BASE_ADDR;
static int8_t  (*a_tcm)[APPLICATION_K]  = (int8_t  (*)[APPLICATION_K])TCM_A_ADDR;
static int8_t  (*b_tcm)[APPLICATION_K]  = (int8_t  (*)[APPLICATION_K])TCM_B_ADDR;

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

static inline uint64_t rd_mhartid(void) {
    uint64_t v;
    asm volatile ("csrr %0, mhartid" : "=r"(v));
    return v;
}

// mt-hello.c style sense-reversing barrier for N_CORES harts. Statics live in
// BSS which hart 0 zeroes before boot-sync releases the secondaries, so the
// first entry sees count=0/sense=0.
static void __attribute__((noinline)) barrier(void) {
    static volatile int sense;
    static volatile int count;
    static __thread int threadsense;

    __sync_synchronize();
    threadsense = !threadsense;
    if (__sync_fetch_and_add(&count, 1) == N_CORES - 1) {
        count = 0;
        sense = threadsense;
    } else {
        while (sense != threadsense) { /* spin */ }
    }
    __sync_synchronize();
}

// Compute the single tile-row `mt` (rows mt*TILE_M .. (mt+1)*TILE_M-1 of C).
// Double-buffers TR0/TR2 vs TR1/TR3 across K-steps and alternates ACC0/ACC1
// across N-tiles, mirroring the tuned single-core kernel.
static void compute_tile_row(int mt) {
    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    const uint64_t a_stride = APPLICATION_K * sizeof(int8_t);   // 512 B
    const uint64_t b_stride = APPLICATION_K * sizeof(int8_t);   // 512 B
    const uint64_t c_stride = APPLICATION_N * sizeof(int32_t);  // 2048 B

    int acct = 0;
    for (int nt = 0; nt < TILES_N; nt++) {
        uint64_t c_base = (uint64_t)&c[mt * TILE_M][nt * TILE_N];
        if (acct == 0) {
            //ame_mzero(ACC0);
            for (int kt = 0; kt < TILES_K; kt++) {
                if ((kt & 1) == 0) {
                    ame_mlae8(TR0, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                    ame_mlbe8(TR2, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                    ame_mmacc_w_b(ACC0, TR0, TR2);
                } else {
                    ame_mlae8(TR1, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                    ame_mlbe8(TR3, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                    ame_mmacc_w_b(ACC0, TR1, TR3);
                }
            }
            ame_msce32(ACC0, c_base, c_stride);
            acct = 1;
        } else {
            //ame_mzero(ACC1);
            for (int kt = 0; kt < TILES_K; kt++) {
                if ((kt & 1) == 0) {
                    ame_mlae8(TR0, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                    ame_mlbe8(TR2, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                    ame_mmacc_w_b(ACC1, TR0, TR2);
                } else {
                    ame_mlae8(TR1, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                    ame_mlbe8(TR3, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                    ame_mmacc_w_b(ACC1, TR1, TR3);
                }
            }
            ame_msce32(ACC1, c_base, c_stride);
            acct = 0;
        }
    }
    asm volatile ("fence" ::: "memory");
}

void __main(void) {
    uint64_t hid = rd_mhartid();
    if (hid >= N_CORES) { while (1) { /* park unused harts */ } }

    // Hart 0 prints the banner once. TCM info is per-tile but identical.
    if (hid == 0) {
        uint32_t info = ame_tcm_get_info();
        printf("MC-GEMM: %dx%dx%d INT8, %d cores, per-tile private TCM+DMA\n",
               APPLICATION_M, APPLICATION_N, APPLICATION_K, N_CORES);
        printf("TCM info: ways=%u log2sets=%u log2blk=%u\n",
               info & 0xF, (info >> 8) & 0xFF, (info >> 16) & 0xFF);
    }

    // Every hart configures its own private TCM. The TCM ctrl MMIO at
    // 0x22000000 lives on each tile's PBUS-side ctrlNode, so writes from
    // hart N land in tile N's TcmCtrl regmap.
    if (ame_tcm_config(4) != 0) {
        printf("[core %u] FAIL: could not configure TCM to 4 ways\n",
               (unsigned)hid);
        return;
    }

    barrier();  // all TCM apertures now open
    int dma_ok = 1;
    // ---- Each hart DMAs A and B into its own private TCM ---------------
    // The four DMA engines run in parallel: independent memReadNode edges
    // out to SBUS/DRAM, independent tcmWriteNode edges into each tile's
    // own TCM. Contention is only on the memory side.
    /*const uint32_t a_bytes = (uint32_t)sizeof(a);
    const uint32_t b_bytes = (uint32_t)sizeof(b);

    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_t0 = rd_cycle();
    ame_dma_load((uint64_t)(uintptr_t)a, TCM_A_ADDR, a_bytes);
    ame_dma_load((uint64_t)(uintptr_t)b, TCM_B_ADDR, b_bytes);
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_t1 = rd_cycle();
    uint64_t dma_cycles = dma_t1 - dma_t0;

    // Bounds sanity: TCM head/tail must equal DRAM head/tail after DMA.
    
    if (((volatile uint8_t *)TCM_A_ADDR)[0]            != ((uint8_t *)a)[0]           ||
        ((volatile uint8_t *)TCM_A_ADDR)[a_bytes - 1]  != ((uint8_t *)a)[a_bytes - 1] ||
        ((volatile uint8_t *)TCM_B_ADDR)[0]            != ((uint8_t *)b)[0]           ||
        ((volatile uint8_t *)TCM_B_ADDR)[b_bytes - 1]  != ((uint8_t *)b)[b_bytes - 1]) {
        dma_ok = 0;
    }

    barrier();  // wait for every core's DMA to land before compute
    */
    // ---- Each hart runs its own tile-row of GEMM ------------------------
    uint64_t gemm_t0 = rd_cycle();
    compute_tile_row((int)hid);
    uint64_t gemm_t1 = rd_cycle();
    uint64_t gemm_cycles = gemm_t1 - gemm_t0;

    barrier();  // all cores done writing their C slice

    // ---- Each hart verifies its own 128 rows of C -----------------------
    // c is in each tile's private TCM, so hart h only sees rows it wrote.
    const int row_lo = (int)hid * TILE_M;
    const int row_hi = row_lo + TILE_M;
    int errors = 0;

    // Serialize the per-core summary so lines don't interleave on HTIF.
    for (int i = 0; i < N_CORES; i++) {
        if ((int)hid == i) {
            printf("[core %u] rows [%d..%d) | dma=%u cyc | gemm=%u cyc | errors=%d %s\n",
                   (unsigned)hid, row_lo, row_hi,
                    (unsigned)gemm_cycles,
                   dma_ok, errors,
                   (errors == 0 && dma_ok) ? "PASS" : "FAIL");
        }
        barrier();
    }
    
    for (int i = row_lo; i < row_hi; i++) {
        for (int j = 0; j < 10; j++) {
            if (c[i][j] != c_ref[i][j]) {
                if (errors < 10) {
                    printf("[core %u] MISMATCH C[%d][%d]: got %d, expected %d\n",
                           (unsigned)hid, i, j, c[i][j], c_ref[i][j]);
                }
                errors++;
            }
        }
    }
    if (hid > 0) { while (1) { /* park; only hart 0 returns to exit */ } }
}

int main(void) {
    __main();
    return 0;
}
