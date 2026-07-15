#include <stdint.h>
#include <stdio.h>

static inline uint64_t read_cycle() {
    uint64_t cycle;
    asm volatile ("rdcycle %0" : "=r" (cycle));
    return cycle;
}

// L1 DCache : 64 sets x 4 ways x 64B = 16KB
// L2 HuanCun: 512 sets x 8 ways x 64B = 256KB
//
// work_array : 32KB  > L1, fits in L2  -- observes L2 hit/miss
// flush_array: 512KB > L2              -- evicts entire L2

#define STRIDE      16       // 16 x 4B = 64B = 1 cache line
#define WS_SIZE     8192     // 8192 x 4B = 32KB
#define FLUSH_SIZE  131072   // 131072 x 4B = 512KB

volatile uint32_t work_array[WS_SIZE];
volatile uint32_t flush_array[FLUSH_SIZE];

static void run_phase(const char *tag, volatile uint32_t *arr, int size, int iters) {
    uint32_t dummy = 0;
    asm volatile ("fence" ::: "memory");
    uint64_t start = read_cycle();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < size; i += STRIDE)
            dummy += arr[i];
    }
    uint64_t cycles   = read_cycle() - start;
    uint64_t accesses = (uint64_t)(size / STRIDE) * iters;
    printf("%-22s %5lu accesses  %8lu cycles  %4lu cyc/line  (dummy=%u)\n",
           tag, accesses, cycles, cycles / accesses, dummy);
    asm volatile ("fence" ::: "memory");
}

int main() {
    printf("=== HuanCun L2 Cache Test ===\n");
    printf("L1=16KB  L2=256KB  WorkSet=32KB  FlushSet=512KB\n\n");

    // Phase 1: cold -- L1 miss + L2 miss, all requests reach LLC
    // Waveform: burst of AcquireBlock from HuanCun toward SBUS
    printf("[Phase 1] Cold read  (L1 miss + L2 miss -> LLC)\n");
    run_phase("phase1_cold", work_array, WS_SIZE, 1);

    // Phase 2: warm -- L1 miss + L2 hit, HuanCun serves all requests
    // Waveform: no traffic on SBUS, HuanCun Grant visible on tile-side TL
    printf("[Phase 2] Warm read  (L1 miss + L2 hit -> HuanCun serves)\n");
    run_phase("phase2_L2hit", work_array, WS_SIZE, 3);

    // Phase 3: flush L2 with a 512KB array, forcing L2 evictions
    // Waveform: interleaved AcquireBlock + ReleaseData on SBUS
    printf("[Phase 3] Flush L2   (512KB working set evicts entire L2)\n");
    run_phase("phase3_flush", flush_array, FLUSH_SIZE, 1);

    // Phase 4: cold again -- work_array evicted in Phase 3, L2 misses again
    // Waveform: same burst pattern as Phase 1
    printf("[Phase 4] Cold again (L2 evicted, should match Phase 1)\n");
    run_phase("phase4_cold_again", work_array, WS_SIZE, 1);

    return 0;
}
