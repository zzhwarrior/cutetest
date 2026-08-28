// Minimal DMA+compute ping-pong skeleton over HuanCun TCM.
// Two 32KB buffers in TCM; DMA prefetches tile N+1 while the current thread
// consumes tile N. Extend `consume_tile()` to plug in Saturn vle / CUTE ame_mlae.

#include <stdint.h>
#include <stdio.h>
#include "ame.h"

#define TILE_BYTES  0x8000UL              // 32 KB per tile
#define BUF_A       0x81000000UL
#define BUF_B       (0x81000000UL + TILE_BYTES)
#define N_TILES     8

static uint8_t __attribute__((aligned(64))) src[N_TILES * TILE_BYTES];

static void init_src(void) {
    for (unsigned i = 0; i < sizeof(src); i++) src[i] = (uint8_t)(i * 7 + 1);
}

// Placeholder: fill in Saturn / CUTE work over `tcm_buf` here.
static uint32_t consume_tile(uint64_t tcm_buf, unsigned tid) {
    volatile uint8_t *p = (volatile uint8_t *)(uintptr_t)tcm_buf;
    uint32_t acc = 0;
    for (unsigned i = 0; i < TILE_BYTES; i += 64) acc += p[i];
    return acc ^ tid;
}

int main(void) {
    init_src();

    // Prefetch tile 0 into BUF_A first.
    ame_dma_load((uint64_t)(uintptr_t)&src[0], BUF_A, TILE_BYTES);

    uint32_t sum = 0;
    for (unsigned t = 0; t < N_TILES; t++) {
        uint64_t cur  = (t & 1) ? BUF_B : BUF_A;
        uint64_t next = (t & 1) ? BUF_A : BUF_B;

        // Kick off DMA for tile t+1 (if any) into the OTHER buffer.
        if (t + 1 < N_TILES) {
            ame_dma_load((uint64_t)(uintptr_t)&src[(t + 1) * TILE_BYTES],
                         next, TILE_BYTES);
        }
        // Compute on the current tile (DMA runs in the background;
        // ame_dma_load is blocking today, so this is sequential
        // until we switch to non-blocking DMA).
        sum += consume_tile(cur, t);
    }
    printf("pingpong sum=0x%x\n", sum);
    return 0;
}
