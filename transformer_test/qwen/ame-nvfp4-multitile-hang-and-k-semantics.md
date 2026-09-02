
**日期**：2026-09-01
**范围**：AME 通路的 NVFP4 scale 引入之后，`ame_nvfp4_gemm_kt.c` 多 K-tile 场景卡死；单 tile phaseD 通过
**结论**：卡死根因不在加速器，在 Shuttle CPU 的 RoCC 派发通路（macro-fusion/rename bug），已通过修改 `ame.h` 的 `AME_ISSUE_WITH_GPR` 宏绕过

---

## 一、Bug 症状

### 现场

- 测试：`/root/CUTE/cutetest/transformer_test/qwen/ame_nvfp4_gemm_kt.c`（APP_M=128, APP_K=256, TILE_K=128, TILES_K=2）
- 单 K-tile phaseD 通过；引入第二个 K-tile 后卡死
- Log 末态（`ZZHDebugEnable` 打印）：

  ```
  [AME-TC<58972>] NOT IDLE: LdEmpty=0 CmpEmpty=0 StEmpty=0 LdFinAll=0 CmpFinAll=0
                  LdIdle=0 CmpIdle=1 StIdle=1
                  | LdH=5 LdT=4 LdFH=4 CmpH=2 CmpT=1 CmpFH=1 StH=1 StT=0
  ```

- Load FIFO 卡在 slot 4（kt=1 的 mlbe4），`Load MicroInst B Finish` 只出现 1 次（kt=0 的 mlbe4），kt=1 那条永远不 finish

### 时间线

| 时刻 | 事件 |
|---|---|
| T=38013 | Load Issue mzero → C Finish @ T=39039 |
| T=39041 | Load Issue mlae4 kt=0 → A Finish @ T=39557 |
| T=39623 | Load Issue mlbe4 kt=0 → **B Finish @ T=40139** |
| T=40205 | Load Issue mlae4 kt=1 + Compute Issue kt=0 → A Finish @ T=40721 |
| T=40787 | Load Issue mlbe4 kt=1 → **B Finish 永不发生** |
| T=41232+ | Compute kt=0 完成，但依赖 mlbe4 kt=1 的 Compute kt=1 卡死 |

---

## 二、真正根因（Shuttle RoCC 派发 bug）

### AMEDecoder 打印的 dispatch 值

```
[AME-DEC 38069] MLAE4  kt=0: A_base=0x81000000  ✓
[AME-DEC 38071] MLBE4  kt=0: B_base=0x81008000  ✓
[AME-DEC 38067] MSET_SCALEB kt=0: rs1=0x81000000  ✗ (应为 0x81011000)
[AME-DEC 38135] MLAE4  kt=1: A_base=0x81000040  ✓
[AME-DEC 38137] MLBE4  kt=1: B_base=0x10201040  ✗ (应为 0x81008040)
```

### Shuttle 自己的 commit trace 却是正确的

```
T=38133  PC=0x800003d8  lui   t1, 0x10201   → 写回 0x10201000  ✓
T=38134  PC=0x800003dc  c.slli t1, t1, 3    → 写回 0x81008000  ✓
T=38134  PC=0x800003de  addi  t1, t1, 64    → 写回 0x81008040  ✓
T=38135  PC=0x800003e2  mlbe4 (RoCC dispatch) → 送出 rs1=0x10201040  ✗
```

三条 ALU 在 Shuttle 自己的 commit 流里全部写对了 t1（终值 `0x81008040`），但**下一拍 mlbe4 通过 RoCC 送给 AME 的 rs1 是 `0x10201040`**——注意这个值 = LUI(`0x10201000`) + ADDI(`+64`)，**中间的 SLLI 被跳过**。

### 判定

- Shuttle 的 commit / 架构寄存器堆 t1 正确
- **RoCC 派发路径读 rs1 时走了另一条通路**，捡到了一份 LUI+ADDI 融合（跳过 SLLI）的旧值
- 是 Shuttle 侧 RoCC dispatch 通路的 macro-fusion / rename bug
- 与 AME、TaskController、scale 通路、依赖跟踪机制**全部无关**

---

## 三、软件绕过

### 修改文件

`/root/CUTE/cutetest/ame_test/ame.h` 中的 `AME_ISSUE_WITH_GPR` 宏

### 修改前（会触发 bug）

```c
#define AME_ISSUE_WITH_GPR(encoding, val_rs1, val_rs2) \
    do { \
        register uint64_t _rs1 __asm__("t1") = (uint64_t)(val_rs1); \
        register uint64_t _rs2 __asm__("t2") = (uint64_t)(val_rs2); \
        __asm__ __volatile__( \
            ".word " GET_VALUE(encoding) "\n\t" \
            : \
            : "r"(_rs1), "r"(_rs2) \
            : "memory" \
        ); \
    } while(0)
```

问题：把 `val_rs1` 直接绑定到 t1，编译器 -O3 下会把 val_rs1 的表达式（LUI+SLLI+ADDI 之类）直接写到 t1，形成三条邻接的 t1-writer，触发 Shuttle 融合 bug。

### 修改后

```c
#define AME_ISSUE_WITH_GPR(encoding, val_rs1, val_rs2) \
    do { \
        __asm__ __volatile__( \
            "mv t1, %0\n\t" \
            "mv t2, %1\n\t" \
            ".word " GET_VALUE(encoding) "\n\t" \
            : \
            : "r"((uint64_t)(val_rs1)), "r"((uint64_t)(val_rs2)) \
            : "t1", "t2", "memory" \
        ); \
    } while(0)
```

关键点：`"r"` 让编译器把 val_rs1 算到**任意一个临时寄存器**（如 a5），最后**用一条独立 `mv` 搬到 t1**。RoCC 读 t1 时其最近 writer 是单一的 `mv`，Shuttle 的融合模式匹配不上。

### 验证反汇编

kt=1 mlbe4 现在的字节流：

```
80000402:  lui   a5, 0x10201    ← 现在往 a5 写，不是 t1
80000406:  slli  a5, a5, 3
80000408:  addi  a5, a5, 64     ← a5 = 0x81008040
8000040c:  mv    t1, a5         ← t1 的唯一 writer
8000040e:  mv    t2, t3
80000410:  mlbe4                ← RoCC 读 t1，拿到 0x81008040
```

### 中间的坑（记录一下走过的弯路）

第一次尝试是**在 RoCC 前面加两个 nop**：

```c
"nop\n\t" \
"nop\n\t" \
".word " GET_VALUE(encoding) "\n\t" \
```

**没用**。因为 Shuttle 的融合发生在**三条 ALU 邻接**这个模式内部，跟后面隔多少 nop 无关。把 t1 的 writer 打散成"lui/slli/addi 到 a5，最后 mv t1, a5"才真正破掉了融合模式。

### Makefile 陷阱

`/root/CUTE/cutetest/transformer_test/qwen/Makefile` 的 `%.o: %.c` 规则没把 `ame.h`（或其他被 include 的头文件）列入依赖，改 header 之后不会触发重编。**改完 ame.h 必须 `rm -f *.o *.riscv` 后 make**。建议后续加自动依赖（`-MMD -MP` + include `.d` 文件），但那是另一件事。

---

## 四、Bug 复现 / 回归检查

### 复现步骤

1. 恢复 `AME_ISSUE_WITH_GPR` 到"直接绑定 t1"的旧版本
2. `cd /root/CUTE/cutetest/transformer_test/qwen && rm -f *.o *.riscv && make ame_nvfp4_gemm_kt.riscv`
3. 反汇编，找 `.word 0xa47301ab`（mlbe4 kt=1）：应看到 `lui t1 / c.slli t1 / addi t1` 三连邻接
4. 跑仿真，日志里 `MLBE4: B_base=0x10201040`（错值），进入 hang
5. 恢复宏为"mv t1, %0"版本，重编，观察 dispatch 值恢复正确

### 回归检查点（未来修改宏时）

- 反汇编所有 `.word 0x[0-9a-f]{4}[03]02b`（AME config）和 `.word 0x[0-9a-f]{4}[03]0ab/1ab/12b/02b`（AME load/store）编码前的字节流
- 确认 t1 的**最后一个 writer** 是一条独立的 `mv` 或 `ld`，而不是 `addi/slli/lui` 之类的 ALU 运算
- 若某处仍触发，可考虑在 `mv t1, %0` 前面加 `fence`（真硬拦，但代价大）

---

## 五、顺带发现的 K 语义矛盾（**尚未修复**）

在排查 hang 的过程中把 `csr_mtilek` 和 K 相关的地方都走了一遍，发现 CUTE 内部**对 "K" 的语义至少有三种互不兼容的定义**共存，短期靠"用户按当前配置手写 K"绕过，长期会咬人。

### 语义 1：RISC-V Matrix Extension Spec v0.6.0

- `mtilek` = K 维**元素数量**
- 元素密度越高（bit 越窄）→ 同一 TR 能装的 K 越多
- Spec 5.2.4：`mtilek ≤ TRLEN / element_bit_width`
- INT8：`mtilek(64)` = 64 个 int8 元素 = 64 字节
- NVFP4：`mtilek(64)` = 64 个 fp4 元素 = 32 字节 = 半个 512-bit RV

### 语义 2：CUTE 当前 `AMEDecoder.scala:138-141`

```scala
csr_mtilek := Mux(rs1_data(...) > ReduceGroupSize.U,
                  ReduceGroupSize.U, rs1_data(...))
```

- `mtilek` 被 clamp 到 `ReduceGroupSize = Tensor_K / ReduceWidthByte`
- 对 `CUTE_4Tops_128SCP`（ReduceWidthByte=64, Tensor_K=64），`ReduceGroupSize = 1`
- 即 `mtilek` 被理解成 **ReduceVector 的数量**，**且**只有 INT8 视角下的换算
- NVFP4 情况下，一个 RV 有 128 个 fp4 元素（512bit/4bit），但 clamp 用的是 INT8 的字节宽度

### 语义 3：`Load_MicroInst` 地址计算 `TaskController.scala:840,845,851`

宏指令通路里 K 通过 `PEDataType.AdataBitWidth(element_type)` 折算：

```scala
Current_Tile_K_Iter * PEDataType.AdataBitWidth / 8 ...
Current_Tile_Tensor_K_Iter * (ReduceWidthByte * ReduceGroupSize)
```

- 这里的 K 又是**字节偏移**语义（`AdataBitWidth/8` = bytes/elem）
- NVFP4 是 sub-byte（4 bit），`AdataBitWidth/8 = 0`，会产生 0 偏移或除零
- 但 AME 通路目前不走 macro-inst decode，所以这个语义没被触发

### 矛盾体现在测试代码里

`ame_nvfp4_smoke_phaseD.c:120-127`（现存注释已经承认了）：

```c
ame_settilek(K);   // K=128
// AMEDecoder clamps csr_mtilek to ReduceGroupSize = Tensor_K/ReduceWidthByte.
// For CUTE_4Tops_128SCP, that's 1. HW then processes exactly one RV
// (= 128 FP4 elements at ReduceWidthByte=64) regardless of the raw K
// requested here. Passing K works because clamp handles the excess.
```

翻译：**软件按 spec 语义传 K=128，硬件按"1 个 RV = INT8-flavored ReduceGroupSize"clamp，恰好因为 NVFP4 一个 RV 是 128 fp4 元素，结果对上了**。

这**不是设计上对齐**，是**巧合对齐**：
- 若 TILE_K 不是 128 的整数倍（比如 spec 允许的 64 或 96），软件按 spec 传的 K 与硬件实际处理的 K 不一致，会静默算错
- 若换配置（例如 `ReduceWidthByte=32`），一个 RV 变成 64 个 fp4，同样一段 C 代码结果就变了
- 若走 macro-inst 通路（`Load_MicroInst.ApplicationTensor_A_Stride` 那种），K 语义又切换到"字节步长"，跟 AME 通路不一致

### `ScaleVecWidth` 也有同样问题 (`CUTEParameters.scala:1160-1170`)

```scala
is (DataTypenvfp4F32)     { scaleVecWidth := (ReduceWidthByte * 8 / 4 / 16).U }
is (DataTypeMxfp8e4m3F32) { scaleVecWidth := (ReduceWidthByte * 8 / 8 / 32).U }
```

- nvfp4: `(64*8)/4/16 = 8`（每 RV 8 字节 scale）
- mxfp8e4m3: `(64*8)/8/32 = 2`（每 RV 2 字节 scale）
- **默认分支返回 0**，其它类型走这条路会得到 `MaxRequestIter = 0`，`AScaleLoader` 状态机在 `MemoryRequestSize=0 === MaxRequestIter=0` 时直接跳过 load，静默走过

`ame_nvfp4_smoke_phaseD.c:25-28` 里有一段"scale 缓冲区填 2KiB 而不是逻辑 512B"的 workaround，就是给 `ScaleLoader over-request` 打的补丁——同一个 K 语义混乱的下游表现。

### 建议的收尾方向（后续做）

按优先级：

1. **短期（不破坏现有测试）**：在 `AMEDecoder` 里把 K 语义**明确记录**成"ReduceGroupSize 数量"，clamp 前**按 element bit width 折算**——`csr_mtilek := min(user_K * element_bits / (ReduceWidthByte*8), ReduceGroupSize)`。让 spec 语义（元素数）和硬件语义（RV 数）在译码点显式转换。

2. **中期**：给 `AdataByteWidth / BdataByteWidth` 补 nvfp4/mxfp4 case（当前默认走 0）。同时用 `bit-width / 8` 而非 `byte-width` 做地址步长，让 sub-byte 类型能走通 macro-inst 通路。

3. **长期**：整个 CUTE 内部把 K 的三种表达（元素数 / RV 数 / 字节偏移）统一到一处，命名区分。当前所有 `K_Iter / Tensor_K_Iter / K_iter_Add` 混用，跟不同调用方约定不同、软件端难写。

这三条跟本次 hang 无关，但都是 scale 通路引入后暴露出来的、随时会咬人的雷。

---

## 六、附录

### 相关文件

- `/root/CUTE/cutetest/ame_test/ame.h`（本次修改）
- `/root/CUTE/cutetest/transformer_test/qwen/ame_nvfp4_gemm_kt.c`（触发测试）
- `/root/CUTE/cutetest/transformer_test/qwen/ame_nvfp4_smoke_phaseD.c`（单 tile 对照，通过）
- `/root/CUTE/src/main/scala/AMEDecoder.scala`（本次未改，之前的 Phase A/B 引入 scale）
- `/root/CUTE/src/main/scala/TaskController.scala`（scale 通路依赖跟踪，已确认无 bug）
- `/root/CUTE/chipyard/sims/verilator/output/chipyard.harness.TestHarness.testL2Dma1core/ame_nvfp4_gemm_kt.out`（诊断用 log）

### 待做（unrelated 但攒着的清洁化）

- `AMEDecoder.scala:370-374` 的 `DataType_A/B/C/D` 应为 byte-width 而非 enum（语义不对但不影响 hang，`Max_Store_Iter` 在 `CDataController` 里是死代码）
- `TaskController.scala:1082-1088` 的 `A/B_SCP_Free` lock 条件应当同时考虑 `Need_Issue_ASL/BSL`，为将来可能的"纯 scale-load"指令预留正确 SCP 语义
- Makefile 补上 header 自动依赖（`-MMD -MP`）

---

## 七、补记：`ame_fence()` clobber 遗漏（Phase E-small 阻塞 bug）

跟第三节的 `AME_ISSUE_WITH_GPR` 是同一家族的问题（内联 asm 与编译器可见性），触发条件完全不同。

### 症状

Phase E-small 三维 tile 测试（`ame_nvfp4_gemm_e256_full.c`，M=N=K=256, 2×2×2 tile）第一次跑：

```
[E256F] corners: C[0][0]=43800000 C[0][255]=43000000 C[255][0]=43800000 C[255][255]=43000000
```

- C[0][0], C[255][0] = 256.0 ✓（nt=0 的 tile）
- C[0][255], C[255][255] = 128.0 ✗（nt=1 的 tile，只有一半 K 贡献）

规律：**只有 nt=1 的 tile 少 1 个 K-tile 的累加**。

### 定位

AME-DEC dispatch trace（截取 mt=0, nt=1 那个 tile）：

```
[AME-DEC 50326] MSET_SCALEA rs1=0x81010000  ✓
[AME-DEC 50328] MSET_SCALEB rs1=0x00000000  ✗ (预期 0x81011400)
[AME-DEC 50330] MLAE4  A=0x81000000  sA=0x81010000
[AME-DEC 50332] MLBE4  B=0x8100c000  sB=0x00000000  ← 用了错的 scale base
...
[AME-DEC 50337] MSET_SCALEB rs1=0x81011c00  ✓ (kt=1 修回来了)
```

只有 nt=1 tile 的 **第一个 kt** 的 mset_scaleb 值是 0，第二个 kt 是对的。

### 反汇编看到

```
80000442:  ame_fence()           ← RoCC 编码 rd=t0, xd=1，返回值写 t0=0
80000446:  ame_mzero(ACC0)
8000044e:  mset_scalea (mv t1, a4)
80000452:  mv t1, t0             ← 编译器以为 t0 还是循环前算好的 0x81011400
80000454:  mv t2, a3
80000456:  mset_scaleb           ← Shuttle 读 t0=0 送给 AME
```

其它 3 处 mset_scaleb 分别用 s4 / s3 / t6 保存的中间值，都没被 fence 清零。**只有恰好被分配到 t0 的那一次**中招。

### 根因

原来的 `ame_fence()`：

```c
static inline void ame_fence(void) {
    __asm__ __volatile__(".word " GET_VALUE(AME_FENCE_M_ENC) "\n\t" ::: "memory");
}
```

`AME_FENCE_M_ENC = ROCC_BIT(0x2B, GPR_T0, 0, 0, 1, 0, 0, 0x70)`——**`rd=t0, xd=1`**，意味着 RoCC dispatch 时会往 rd（t0）写回一个值（fence 场景下永远是 0，用户从来不读）。

问题：inline asm 的 clobber 列表**只声明了 `"memory"`，没声明 `"t0"`**。编译器就以为 `ame_fence()` 不动 t0，可以在 fence 前把长期活跃的临时值放进 t0 缓存，fence 后继续用——结果值被 RoCC 写回清成 0。

### 修法

```c
static inline void ame_fence(void) {
    __asm__ __volatile__(".word " GET_VALUE(AME_FENCE_M_ENC) "\n\t" ::: "t0", "memory");
}
```

clobber 加 `"t0"`，编译器就不再把值缓存在 t0 里跨 fence。修完 e256_full PASS，`C[i][j] = 0x43800000 = 256.0` 全对。

### 回归检查

反汇编 mset_scaleb（编码 `0x4203002b`）前的 `mv t1, ??` 序列，**不应出现 `mv t1, t0`**（除非那一处代码逻辑上就是取 fence 之后 t0 里的东西，本代码库里不存在这种用法）。同类 grep：所有 AME_ISSUE 前的 `mv t1, t0` / `mv t2, t0` 都要避免。

### 与第三节的关系

- 第三节：`AME_ISSUE_WITH_GPR` 让编译器把 LUI+SLLI+ADDI 直接算到 t1，触发 **Shuttle CPU** 的 RoCC 派发通路的融合 bug。修法：拆成"scratch reg → mv t1"。
- 第七节：`ame_fence` clobber 漏声明 t0，触发 **编译器**假定 t0 跨 fence 保值。修法：加 clobber。

两个 bug 都属于"内联 asm 与编译器/CPU 契约漏洞"，都表现为 rs1 送到 AME 的值和源码语义不一致。**排查方向都是从 `AME-DEC ... rs1=<wrong value>` 打印倒查反汇编**。

### 补：qwen 的 "bf16" 命名其实是 IEEE fp16（Phase G.2 发现）

`qwen_full_attn_nvfp4.c` 里的 `fuse_nvfp4_bf16_cvrt`、`q_buf_bf16`、`k_buf_bf16`、`matmul_cute_bf16` 等命名带 "bf16"，但实际做的是 **IEEE fp16**：

- `-march=rv64imafdcv_zvfh_zfh` 只启用 `zvfh`（IEEE fp16 vector 扩展），没有 bf16 扩展
- `_Float16` 在这套工具链下 = **IEEE fp16**（不是 bf16）
- `__riscv_vfncvt_f_f_w_f16m2` 是 fp32 → **IEEE fp16**，不是 fp32 → bf16

Phase G.2 独立测试用 A=B=1.0 输入，跑 fuse_nvfp4_bf16_cvrt 后写出的值是 `0x5C00` = IEEE fp16(256.0)，不是 `0x4380` = bf16(256.0)。

**验证**：
```
fp16(256.0): sign=0, exp=8+15=23=10111, mantissa=0 → 0 10111 0000000000 = 0x5C00 ✓
bf16(256.0): fp32(256.0)=0x43800000 的高 16 位 = 0x4380
```

**潜在的 qwen 数值 bug**：qwen 的 score matmul 用 `matmul_cute_bf16`（数据类型 `CUTEDataTypeBF16BF16F32`，硬件真的按 bf16 处理），但输入 buffer `q_buf_bf16` 里存的是 IEEE fp16 值。fp16 位模式被当 bf16 解读会得到非常离谱的数值（例如 fp16(256)=0x5C00 → bf16 解读=1.44×10^17）。qwen 端到端如果没做逐元素对拍就发现不了。

**对 Phase G 集成的影响**：
- 保持行为一致：AME 版的 wrapper 也照抄 qwen 的错误命名和实际 fp16 转换。这样 qwen 的整体数值流（含 bug）在 AME 版和 CUTE 版之间是**同样错的**，两者可以对拍。
- 修 qwen bug 是**独立事项**：要么把 fuse 改成真正的 bf16（需要 bf16 RVV 扩展或标量循环），要么把 score matmul 改成 fp16（`mfmacc.s.h`）。Phase H 或以后再动。

### 补：`asm volatile fence rw, rw` **不能**替代 `ame_fence()`

Phase G.2 尝试用 CPU 内存 fence 替换 wrapper 里的 per-tile `ame_fence()` 以省 wait 成本，直接失败：

```
[G2 TO_F32] FAIL: 1280 wrong; first C[0][0]=0x43600000 (=224.0) expected 0x43800000 (=256.0)
```

224 = 256 × 7/8 —— msce32 的 TL 写只完成 7/8，CPU 就读 staging 走 fuse-op。

原因：`asm volatile fence rw, rw ::: memory` 是 **CPU 内部** 的内存序 barrier，只顺序 CPU 自己的 load/store。AME 是**独立 agent**，它的 msce32 → CML → TL A-channel 写 TCM 是异步的，CPU fence 感知不到 TL D-ack 是否回来。

只有 `ame_fence()` 靠 RoCC `is_ame_fence` 拦住 CPU commit，直到 TaskController.io.ame_all_idle（所有 3 个 issue state machine idle + FIFO 空）才放行——这时 CML 的 Store 已经完整 ACK 完成，TCM 内容已确定。

**结论**：wrapper 里 per-tile 同步必须用 `ame_fence()`，不能用 CPU fence。除非将来加了每指令粒度的 completion tracking（例如让 msce32 返回一个 task ID + `CUTE_TASK_END`），否则 fence-per-tile 是必需的。

### 其它 RoCC 指令 clobber 巡检

对 `ame.h` 里所有 `xd=1` 的 RoCC 编码都要过一遍：

| 指令 | 编码 | xd | rd | 现状 |
|---|---|---|---|---|
| `ame_fence` | `AME_FENCE_M_ENC` | 1 | t0 | ✅ 已加 `"t0"` clobber |
| `ame_status` | `AME_MSTATUS_ENC` | 1 | t0 | ✅ 已有 `"t0"` clobber（也通过 mv 出到 output） |
| `ame_dma_load` | `AME_DMA_LOAD_ENC` | 0 | 0 | ✅ 不写回，安全 |
| 其它 `AME_ISSUE*` | 通过 `.word` 直接发的 config/load/store/matmul | 0 | 0 | ✅ 不写回，安全 |

---

## 八、GCC RVV codegen 在多 stage softmax 上的顽固误编（Phase G.4 阻塞 bug）

**Toolchain**：`riscv64-unknown-elf-gcc 15.1.0`
**Flags**：`-O3 -march=rv64imafdcv_zvfh_zfh -mabi=lp64d -specs=htif_nano.specs`
**触发代码**：`fuse_masked_softmax_bf16` （原始版本见 `qwen_full_attn_nvfp4.c:497-559`）

跟前面所有 bug 不同——这个是**GCC 自己的 RVV codegen 有多个 bug**，跟硬件、跟 CPU、跟内联汇编约定都无关。是纯软件端的坑。

### Softmax 是什么（放在这里方便回头查）

Softmax 对一个 vector `x = [x₀..x_{N-1}]` 输出同长度 vector `y`：

```
y_i = exp(x_i) / Σ_j exp(x_j)
```

直接算会溢出（exp(50) 就爆 fp32）。**Numerically-stable softmax** 先减 max：

```
m = max(x)
y_i = exp(x_i - m) / Σ_j exp(x_j - m)
```

减 max 后所有 exp 输入 ≤ 0，结果都在 (0, 1] 之间，安全。

Qwen attention 里加了 scale 和 mask：

```
x_i *= 1/√d                    (INV_SQRT_HDIM = 1/16 for HEAD_DIM=256)
if !mask[i]: x_i = -∞          (因果 mask 遮住未来 token)
m = max(x_i)                    (masked max)
sum = Σ_i exp(x_i - m)          (masked cell 贡献 0)
y_i = exp(x_i-m) / sum → fp16   (存回 attention score buffer)
```

我们的测试：`dim_j = N = 128`（一 row 128 个 score），`dim_i = M = 128` rows。

### 为什么必须拆成 4 个 stage

要算 softmax 需要**两次 pass**——第一次要先扫全 row 得到 max 和 sum，第二次才能算 normalized output。qwen 实现拆成 4 个 stage（每个 stage 扫 row 一次）：

| Stage | 目的 | 每 cell 做的事 | Load | Store |
|---|---|---|---|---|
| **1** | scale | `x[k] *= 1/16` | `x[k]` | `x[k]` |
| **2** | masked max | `m = max(mask ? x[k] : -∞)` | `x[k], mask[k]` | 累加器 (寄存器) |
| **3** | exp + sum | `x[k] = exp(x[k]-m); s += x[k]` | `x[k], mask[k]` | `x[k]` + 累加器 |
| **4** | normalize + cvrt | `y[k] = fp16(x[k] / s)` | `x[k]` | `y[k]` (fp16) |

Stage 1 / 4 是简单 element-wise 算术；stage 2 有 reduction (max)；**stage 3 最复杂**——三件事：exp 计算、写回、accumulate。

### RVV 怎么向量化每个 stage

RVV 每条指令一次处理 `vl` 个元素（vlmax 取决于 VLEN、SEW、LMUL，对 e32m4 通常是 32/64）。

**Stage 1**（3 条 RVV 指令 × 128/vl 次迭代）：
```
vle32 → vfmul.vf → vse32
```

**Stage 2**（reduction）：
```
vfmv.v.f -INFINITY 初始化累加器
loop:  vle32 → vlm → vmerge → vfmax
vfredmax 归约成标量 max_val
```

**Stage 3**（bug 出的地方）：
```
vfmv.v.f 0.0 初始化 sum
loop:  vle32 → vfsub.vf → vlm → vmerge → vec_exp(内部 ~15 条 RVV) → vse32 → vfadd
vfredusum 归约成标量 sum_exp
```

**Stage 4**（fp32 → fp16 narrow cvrt）：
```
vle32 → vfmul.vf → vfncvt.f.f.w → vse16
```

### Bug 精确到"哪一步"

**不是**所有 RVV softmax 命令都错——**只有 stage 3 那一段**。

证据：
1. Stage 1/2/4 单独看没问题。同样的 RVV 模式在别处也用（比如 `matmul_ame_bf16` GEMM 主体，e256_full/e512 都 PASS），都工作正常
2. Stage 3 的**独有组合**：
   - `vec_exp` 被 inline 展开成 ~15 条 RVV 指令
   - masked accumulation (`vmerge` + `vfadd`)
   - **同一内存地址 `row_x` 在同一 stage 内被 vle32 读 + vse32 写**（read-modify-write 同址）
   - 累加器 `sum_vec` 跨 iteration 保留
3. Stage 1 是"读 x → 乘 → 写 x"，虽然也是同址 read-modify-write，但**每个元素只碰一次，没有累加器**——GCC 找不到重排空间
4. Stage 2 只 vle32 读，没 vse32 写——不存在同址读写问题
5. Stage 4 vle32 读 x + vse16 写 y——**不同的内存位置**，也没同址问题

结论：**bug 触发条件 = `read row_x + 长指令序列（vec_exp）+ write row_x + accumulate` 全都在一个 stage 里**。三个条件缺一不可。

### 为什么"只 row 0 (i=0) 错"

Outer loop 是 `for (i = 0; i < dim_i; i++)`。row 0 是**第一次**进入 stage 3。i=1..127 时前一次 iter 已经运行完，把某些寄存器/vtype 状态设成了"合适"的值，stage 3 就工作。i=0 时这些状态是**函数入口的初始态**（可能是上一个 stage 遗留、或者是未初始化的 vtype）。

具体是哪个状态没设对？反汇编看不出来——RVV codegen 在这个函数展开成几百条向量指令，微妙的错位无法目视定位。加 debug 打印会当作 memory clobber 屏障，阻止 GCC 做那些错误的优化——结果 i=0 也对。这是**间接**证据。

### 为什么 debug printf 会"意外治好" bug

`printf` 是外部函数调用，GCC 视其为 **memory clobber** + **不可预测副作用**。这打断了 GCC 的两个关键优化：
- **Software pipelining**：把 iter i 的 stage 4 和 iter i+1 的 stage 1 重排交叠
- **Cross-stage 寄存器复用**：把 stage 2/3/4 用的 v-register 复用同一组物理寄存器

一旦 printf 挡在中间，GCC 只好"老老实实按顺序执行"——bug 消失。

所以看到 **"加 debug 就对，去掉就错"** 是编译器 bug 的强信号，也是这个 bug 唯一能定位到的方式（因为反汇编太长看不清）。

### 为什么 debug 打印展示 sum_exp = 128 但输出按 sum = 96 算

诊断打印显示：
```
[softmax DBG] i=0 row_x[0..7] = 16.0 均匀  (stage 1 后正确)
[softmax DBG] i=0 max_val = 16.0            (stage 2 正确)
[softmax DBG] i=0 sum_exp = 128.0           (stage 3 内部计算正确)
```

sum_exp = 128 说明 stage 3 **在寄存器里**算的和是对的。但最终 stage 4 写出的 C[0][j] 按 sum=96 算——说明 GCC **在 stage 3 → stage 4 之间**做了非法优化，让 stage 4 看到的 row_x 状态跟 stage 3 声称写入的不一致。

具体是什么：**GCC 可能没执行 stage 3 的部分 vse32**（把它认作 dead store，因为紧跟着 stage 4 也要写 row_x……不，stage 4 只 vfmul→vfncvt→vse16 到 row_y，并没有写 row_x）。或者 **GCC 把 stage 3 的 vfadd（累加）和 vse32（写回）拆分调度到错误位置**。

反正结果是：**stage 3 输出到内存的 row_x 缺了 32 个 cell 的值**（保留了 stage 1 的 16.0 而不是 exp(0)=1.0）。stage 4 读 row_x 时看到 [1.0×96, 16.0×32]，乘以 inv_sum = 1/128 得到 [1/128×96, 16/128×32] 混合值——但因为 max 是 16 而不是 1，softmax 归一化"以为"分母只有 96 个 1.0，得到 1/96。

### 症状矩阵

用 `ame_bf16_wrapper_g4_test.c` 复现（M=N=128, K=256, 单 tile, A=B=1.0, all-1 mask）：

- **期望**：每 cell 输出 fp16(1/128) = `0x2000`
- **实际（因 GCC 优化等级而异）**：

| Opt level | C[0][0] | C[0][127] | C[127][127] | 说明 |
|---|---|---|---|---|
| `-O3` 默认 | `0x2155` | `0x2155` | `0x2000` ✓ | fp16(1/96)，row 0 sum_exp=96 而不是 128 |
| `-O2` 局部 | `0x2155` | `0x2155` | `0x2000` ✓ | 同上 |
| `-O1` 局部 | `0x0000` | `0x7e00` | `0x2000` ✓ | row 0 变成 0 和 NaN 混合 |
| `-O0` 局部 | `0x2000` ✓ | `0x2000` ✓ | `0x2000` ✓ | 正确，但慢约 8× |

**唯一稳定通过的路径是**：整个 fuse-op 函数 `#pragma GCC optimize("O0")` 或替换成**纯标量实现**。

### Bug 特征

1. **只影响 `i=0`**（outer softmax loop 的第一次迭代）
2. **其它 rows 1..127 都正确**
3. **不同 opt level 给出不同的错误值**，说明 GCC 的多个优化 pass 都会在这个模式上出错
4. **和 debug printf 有交互**：加打印会改变 pattern，去掉又变。printf 的 memory clobber 意外地当了 barrier

### 排查过程

按下面这个顺序尝试了每个"标准"的隔离手段，**全部无效**：

**a) block-scope 每个 stage 的 `vl` 变量**（第 3.3 节）
- 原来一份 `size_t vl` 跨 4 个 stage 共用，怀疑寄存器分配复用出问题
- 反汇编验证：修完后 `subw avl, avl, vl_reg` 使用了 vsetvli 返回的实际 vl，不再是错误的 dim_j
- **结果**：还是 FAIL，row 0 仍然 `0x2155`

**b) 每个 stage 后加 `asm volatile ("" ::: "memory")`**
- 强迫编译器保持 stage 顺序，禁止跨 stage 优化
- **结果**：还是 FAIL

**c) staging buffer 从 TCM (0x81100000) 挪到 DRAM (.bss 静态数组)**
- 排除 TCM/L1 一致性可能
- **结果**：还是 FAIL

**d) `__attribute__((noinline))` on fuse function**
- 防止 fuse-op 被 inline 到 wrapper 里跨函数优化
- **结果**：还是 FAIL

**e) `#pragma GCC optimize("O2")` 局部降级**
- **结果**：FAIL（`0x2155`）

**f) `#pragma GCC optimize("O1")` 局部降级**
- **结果**：FAIL，但错法不一样（`0x0000` / `0x7e00` NaN 混合）

**g) `#pragma GCC optimize("O0")` 局部降级**
- **结果**：PASS ✓，但 cycles 从 O3 的 ~76k 涨到 ~640k（8×）

**h) 纯标量重写（最终采用）**
- 不用任何 RVV intrinsic
- 用手写的 Chebyshev 多项式 `scalar_exp()`（复用 `vec_exp` 的系数），因为 `htif_nano` libc 没 `expf`
- **结果**：PASS ✓，cycles 估计 200-500k

### 为什么值是 `0x2155`

一个 128 cell 的 row 全 1.0 的 softmax 分母应该是 128，softmax 输出 = `1/128 = 2^-7 = 0x2000`。

`0x2155` 解码 = `1.3330 × 2^-7 = 1/96`。即分母是 96 不是 128。**32 个 cell 没贡献到 sum。**

诊断打印证实：
- Row 0 stage 1 输出 `row_x[0..7] = 0x41800000` (16.0 均匀)  ✓
- Row 0 stage 2 max_val = `0x41800000` (16.0)  ✓
- Row 0 stage 3 sum_exp = `0x43000000` (128.0)  ✓  ← 与 0x2155 表现矛盾
- 但**最终写出的 C[0][j] 仍然是 `0x2155`**

即 stage 3 内部计算是对的（sum=128），可是 stage 4 输出却按 sum=96 算。GCC 在 stage 3→stage 4 之间做了某种非法优化（可能是软流水化、循环融合，或跨 outer-iteration 的向量寄存器复用），把 row 0 的 stage 3 输出提前"预测"成了 row_x 中还有 32 个未被覆盖的 stage-1 值。

反汇编看不出简洁的错点——不像第三节的 LUI+SLLI+ADDI 或第七节的 clobber 那样一目了然。RVV codegen 在这个函数生成了几百条向量指令，微妙的错位无法目视定位。

### 最终修法（scalar fallback）

```c
static inline float scalar_exp(float x) {
    // Chebyshev 多项式版 exp，直接复用 vec_exp 的系数
    const float NEG_LN2 = -0.69314718056f;
    const float INV_LN2 = 1.44269504089f;
    float af = x * INV_LN2;
    int32_t a = (int32_t)(af + (af >= 0.0f ? 0.5f : -0.5f));
    if (a > 127)  return INFINITY;
    if (a < -126) return 0.0f;
    uint32_t bits = (uint32_t)(a + 127) << 23;
    float a2; memcpy(&a2, &bits, sizeof(a2));
    float b = x + NEG_LN2 * (float)a;
    float p = 0.001388888889f;
    p = p * b + 0.008333333333f;
    p = p * b + 0.041666666667f;
    p = p * b + 0.166666666667f;
    p = p * b + 0.5f;
    p = p * b + 1.0f;
    p = p * b + 1.0f;
    return a2 * p;
}

static void __attribute__((noinline))
fuse_masked_softmax_bf16(void *input, void *output, ...,
                         int dim_i, int dim_j, ...) {
    for (int i = 0; i < dim_i; i++) {
        // stage 1: scale by INV_SQRT_HDIM  (scalar loop)
        // stage 2: masked max               (scalar loop)
        // stage 3: exp(x-max), sum          (scalar loop, uses scalar_exp)
        // stage 4: normalize + fp32→fp16    (scalar loop, C-cast)
    }
}
```

- 完整 O(N²) 标量循环，无 RVV intrinsic
- `scalar_exp(0)` 精确等于 1.0（Chebyshev polynomial 在 b=0 时 p = c0 = 1.0）
- 对 x=0（uniform row）行为跟 vec_exp 逐位一致

### 性能影响

Softmax 是 O(N²)，GEMM 是 O(N³)。在 Qwen 一层：
- 16 个 score matmul，每个 128×128×256 = 4.2M muladd
- 16 个 softmax，每个 128×128 = 16k muladd × 4 stages ≈ 65k 标量 op

Softmax 标量版比 RVV 版慢 5-10×，但绝对时间跟 GEMM 比可忽略（估算 <5% 总耗时）。

### 回归检查

如果哪天 GCC 修了这个 bug，可以尝试恢复 RVV 版本。判据：
1. 用同样测试跑一遍，row 0 的 C[0][0..127] 全等于 `0x2000`
2. 反汇编 fuse function，确认 stage 3 循环用 `vsetvli vl_reg, avl, ...` 返回实际 vl，且 subw 用该 vl_reg
3. 尝试删掉 `__attribute__((noinline))`，看内联进 wrapper 后是否还对

### 上游影响

`qwen_full_attn_nvfp4.c`（CUTE 版）里的 `fuse_masked_softmax_bf16` 是**同一份 RVV 代码**，所以理论上也会 miscompile。之所以没被发现，是因为原始 Qwen 测试只测 cycle 数，不做逐元素数值对拍。**如果哪天 Qwen 数值也要验证**，这个 fuse 也需要同样换成标量版。Phase G.5 集成 `qwen_full_attn_nvfp4_ame.c` 时同步修复。

### 与前几节的关系

- 第三节 (`AME_ISSUE_WITH_GPR`)、第七节 (`ame_fence`)：**内联 asm 与编译器契约** 问题，症状是 rs1 送到 AME 的值错
- 本节 (softmax)：**纯 RVV intrinsic codegen** 问题，症状是 CPU 端计算结果错
- 共同点：都是编译器层面的 bug/限制，都不是硬件问题。**排查 GEMM 端到端出错时，除了怀疑硬件/RTL，也要考虑编译器坑**
- 教训：所有涉及 RVV 的复杂多 stage 代码（softmax、layernorm、attention 等），**先用 scalar 版对拍数值正确性**，再换成 RVV 加速。永远保留 scalar 参考实现
