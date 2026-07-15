// AME + RVV 协同样例：Linear + RMSNorm
//
// 计算流程：
//   Step 1 (AME): Y[M][N] = X[M][K] × W[N][K]^T   (INT8 GEMM, 结果 INT32)
//   Step 2 (RVV): 将 INT32 结果转为 FP32，计算 RMSNorm
//                 rms = sqrt(mean(Y^2) + eps)
//                 out[i] = Y[i] / rms * gamma[i]
//
// 硬件配置假设：TILE_M=128, TILE_N=128, TILE_K=64
// 本例简化为单 tile：M=128, N=128, K=64
//
// AME 与 RVV 的协作方式：
//   - AME 异步执行 GEMM，CPU 继续配置 RVV
//   - fence.m 同步点：等待 AME 完成后 RVV 才读结果
//   - RVV 直接操作 AME store 到内存的结果

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "ame.h"

#define M       128
#define N       128
#define K       64
#define TILE_M  128
#define TILE_N  128
#define TILE_K  64
#define EPS     1e-6f

// 输入输出数据（静态分配，对齐到 64 字节）
static int8_t  X[M][K]   __attribute__((aligned(64)));
static int8_t  W[N][K]   __attribute__((aligned(64)));  // W 已转置存储
static int32_t Y[M][N]   __attribute__((aligned(64)));  // AME GEMM 输出
static float   out[M][N] __attribute__((aligned(64)));  // RMSNorm 输出
static float   gamma_w[N] __attribute__((aligned(64))); // RMSNorm 权重

// ============================================================
// RVV 内联函数：INT32 → FP32 转换 + RMSNorm（逐行处理）
// 每行 N=128 个元素，VLEN=512bit → 每次处理 16 个 FP32
// ============================================================

static void rvv_rmsnorm_row(const int32_t* y_row, const float* gamma,
                             float* out_row, int n, float eps)
{
    // Step A: 计算 sum of squares（INT32 → FP32 累加）
    float sum_sq = 0.0f;
    int i = 0;

    // RVV: 向量化累加 sum(y^2)
    // vsetvli 设置向量长度，vle32 加载，vfcvt 转换，vfmul 平方，vfredusum 规约
    asm volatile (
        "vsetvli    zero, %[n], e32, m4, ta, ma\n\t"  // 设置 vl，e32=32bit元素，m4=4倍寄存器组
        "vmv.v.x    v28, zero\n\t"                     // v28 = 0 (accumulator)
        :
        : [n] "r" (n)
        : "memory"
    );

    while (i < n) {
        int vl;
        asm volatile (
            "vsetvli    %[vl], %[rem], e32, m4, ta, ma\n\t"
            "vle32.v    v0, (%[src])\n\t"              // 加载 INT32
            "vfcvt.f.x.v v4, v0\n\t"                  // INT32 → FP32
            "vfmul.vv   v8, v4, v4\n\t"               // y^2
            "vfadd.vv   v28, v28, v8\n\t"             // 累加
            : [vl] "=r" (vl)
            : [rem] "r" (n - i), [src] "r" (y_row + i)
            : "memory"
        );
        i += vl;
    }

    // 规约：向量内求和
    asm volatile (
        "vsetvli    zero, %[n], e32, m4, ta, ma\n\t"
        "vmv.s.x    v0, zero\n\t"                      // 标量初始化为 0
        "vfredusum.vs v0, v28, v0\n\t"                 // 规约求和 → v0[0]
        "vfmv.f.s   %[res], v0\n\t"                    // 取出标量结果
        : [res] "=f" (sum_sq)
        : [n] "r" (n)
        : "memory"
    );

    // Step B: 计算 rms = 1 / sqrt(mean(y^2) + eps)
    float rms_inv = 1.0f / sqrtf(sum_sq / n + eps);

    // Step C: out[i] = (y[i] / rms) * gamma[i]
    // RVV: 向量化 scale
    i = 0;
    while (i < n) {
        int vl;
        asm volatile (
            "vsetvli    %[vl], %[rem], e32, m4, ta, ma\n\t"
            "vle32.v    v0, (%[src])\n\t"              // 加载 INT32
            "vfcvt.f.x.v v4, v0\n\t"                  // INT32 → FP32
            "vfmul.vf   v4, v4, %[rms]\n\t"           // * rms_inv
            "vle32.v    v8, (%[gam])\n\t"              // 加载 gamma
            "vfmul.vv   v4, v4, v8\n\t"               // * gamma
            "vse32.v    v4, (%[dst])\n\t"              // 写出
            : [vl] "=r" (vl)
            : [rem] "r" (n - i),
              [src] "r" (y_row + i),
              [gam] "r" (gamma + i),
              [dst] "r" (out_row + i),
              [rms] "f" (rms_inv)
            : "memory"
        );
        i += vl;
    }
}

// ============================================================
// 初始化测试数据
// ============================================================
static void init_data(void)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            X[i][j] = (int8_t)((i + j) % 127);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            W[i][j] = (int8_t)((i * 3 + j) % 127);

    for (int i = 0; i < N; i++)
        gamma_w[i] = 1.0f;  // gamma 全 1，便于验证
}

// ============================================================
// 主函数
// ============================================================
int main(void)
{
    printf("AME+RVV Linear+RMSNorm: M=%d N=%d K=%d\n", M, N, K);

    init_data();

    // --- 配置 AME tile 尺寸 ---
    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    uint64_t a_stride = K * sizeof(int8_t);   // X 行步长
    uint64_t b_stride = K * sizeof(int8_t);   // W 行步长（已转置存储）
    uint64_t c_stride = N * sizeof(int32_t);  // Y 行步长

    uint64_t t_start, t_gemm_done, t_norm_done;
    asm volatile("rdcycle %0" : "=r"(t_start));

    // ============================================================
    // Step 1: AME 执行 GEMM（异步，CPU 不等待）
    // Y[128][128] = X[128][64] × W[128][64]^T
    // ============================================================
    ame_mzero(ACC0);

    // 单 tile，K 方向只有一步（TILE_K=64=K）
    // 使用 bank0：TR0(A), TR2(B), ACC0(C)
    ame_mlae8(TR0, (uint64_t)X, a_stride);
    ame_mlbe8(TR2, (uint64_t)W, b_stride);
    ame_mmacc_w_b(ACC0, TR0, TR2);
    ame_msce32(ACC0, (uint64_t)Y, c_stride);

    // ============================================================
    // fence.m：等待 AME 全部完成
    // io.busy 拉高，CPU pipeline stall，直到 ame_all_idle
    // ============================================================
    ame_fence();

    asm volatile("rdcycle %0" : "=r"(t_gemm_done));

    // ============================================================
    // Step 2: RVV 执行 RMSNorm（逐行）
    // 此时 Y 已在内存中，RVV 直接读取
    // ============================================================
    for (int i = 0; i < M; i++) {
        rvv_rmsnorm_row(Y[i], gamma_w, out[i], N, EPS);
    }

    asm volatile("rdcycle %0" : "=r"(t_norm_done));

    // ============================================================
    // 验证：打印前 4 行前 4 列
    // ============================================================
    printf("Y (INT32, first 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%8d ", Y[i][j]);
        printf("\n");
    }

    printf("out (FP32 RMSNorm, first 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%8.4f ", out[i][j]);
        printf("\n");
    }

    printf("Cycles: GEMM=%lu, RMSNorm=%lu, Total=%lu\n",
           t_gemm_done - t_start,
           t_norm_done - t_gemm_done,
           t_norm_done - t_start);

    return 0;
}
