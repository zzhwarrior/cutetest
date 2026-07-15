// Generate matmul test data header file
// Usage: ./gen_matmul_value M N K [seed]
// Output: matmul_value_mnk_M_N_K.h with a[M][K], b[N][K], c address

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s M N K [seed] [c_base_addr]\n", argv[0]);
        fprintf(stderr, "  M: rows of A and C\n");
        fprintf(stderr, "  N: cols of C (rows of B, since C = A * B^T)\n");
        fprintf(stderr, "  K: cols of A and B\n");
        fprintf(stderr, "  seed: random seed (default: 42)\n");
        fprintf(stderr, "  c_base_addr: hex address for C matrix (default: 0x80020000)\n");
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    unsigned int seed = (argc > 4) ? atoi(argv[4]) : 42;
    unsigned long c_base = (argc > 5) ? strtoul(argv[5], NULL, 16) : 0x81000000UL;

    srand(seed);

    char filename[256];
    snprintf(filename, sizeof(filename), "matmul_value_mnk_%d_%d_%d.h", M, N, K);

    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fprintf(f, "#ifndef MATMUL_VALUE_MNK_%d_%d_%d_H\n", M, N, K);
    fprintf(f, "#define MATMUL_VALUE_MNK_%d_%d_%d_H\n\n", M, N, K);
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "#define APPLICATION_M %d\n", M);
    fprintf(f, "#define APPLICATION_N %d\n", N);
    fprintf(f, "#define APPLICATION_K %d\n\n", K);

    // Generate A[M][K]
    fprintf(f, "// A[%d][%d] int8, row-major\n", M, K);
    fprintf(f, "static int8_t a[%d][%d] __attribute__((aligned(256))) = {\n", M, K);
    for (int i = 0; i < M; i++) {
        fprintf(f, "{");
        for (int j = 0; j < K; j++) {
            int val = (rand() % 256) - 128;
            fprintf(f, "%d", val);
            if (j < K - 1) fprintf(f, ",");
        }
        fprintf(f, "},\n");
    }
    fprintf(f, "};\n\n");

    // Generate B[N][K]
    fprintf(f, "// B[%d][%d] int8, row-major (B^T used in matmul: C = A * B^T)\n", N, K);
    fprintf(f, "static int8_t b[%d][%d] __attribute__((aligned(256))) = {\n", N, K);
    for (int i = 0; i < N; i++) {
        fprintf(f, "{");
        for (int j = 0; j < K; j++) {
            int val = (rand() % 256) - 128;
            fprintf(f, "%d", val);
            if (j < K - 1) fprintf(f, ",");
        }
        fprintf(f, "},\n");
    }
    fprintf(f, "};\n\n");

    fprintf(f, "#endif\n");
    fclose(f);

    printf("Generated %s (M=%d, N=%d, K=%d, seed=%u, c_base=0x%lx)\n",
           filename, M, N, K, seed, c_base);
    printf("  A: %d bytes, B: %d bytes, C: %d bytes\n",
           M * K, N * K, M * N * 4);

    return 0;
}
