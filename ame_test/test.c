#include <stdint.h>
#include <stdio.h>

static inline uint64_t read_cycle() {
    uint64_t cycle;
    asm volatile ("rdcycle %0" : "=r" (cycle));
    return cycle;
}

// 32KB working set: fits in L2 (256KB) but NOT in L1 DCache (16KB)
#define ARRAY_SIZE 8192   // 8192 x 4B = 32KB
#define STRIDE     16     // 16 x 4B = 64B = 1 cache line
#define ITERATIONS 5

volatile uint32_t test_array[ARRAY_SIZE];

int main() {
    uint64_t start_cycle, end_cycle;
    uint32_t dummy = 0;

    printf("=== Phase 1: Warmup (fills cache hierarchy) ===\n");
    uint64_t warmup_start = read_cycle();
    for (int i = 0; i < ARRAY_SIZE; i += STRIDE) {
        test_array[i] = i;
    }
    uint64_t warmup_cycles = read_cycle() - warmup_start;
    uint64_t warmup_accesses = ARRAY_SIZE / STRIDE;
    printf("Warmup accesses: %lu, cycles: %lu, avg: %lu cyc/line\n",
           warmup_accesses, warmup_cycles, warmup_cycles / warmup_accesses);

    // fence: ensure all warmup writes are visible before measurement
    asm volatile ("fence" ::: "memory");

    printf("=== Phase 2: Measurement (should hit in cache if L2 works) ===\n");
    start_cycle = read_cycle();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < ARRAY_SIZE; i += STRIDE) {
            dummy += test_array[i];
        }
    }
    end_cycle = read_cycle();

    uint64_t total_cycles   = end_cycle - start_cycle;
    uint64_t total_accesses = (uint64_t)(ARRAY_SIZE / STRIDE) * ITERATIONS;
    uint64_t cycles_per_access = total_cycles / total_accesses;

    printf("Total Accesses: %lu\n", total_accesses);
    printf("Total Cycles: %lu\n", total_cycles);
    printf("Average Cycles per Cache Line Access: %lu\n", cycles_per_access);
    printf("Dummy value (ignore): %u\n", dummy);

    return 0;
}
