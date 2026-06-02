#include <stdio.h>
// #include <riscv-pk/encoding.h>
// #include <riscv-pk/marchid.h>
#include "marchid.h"
#include <stdint.h>
#include "cuteMarcoinstHelper.h"
#include "conv_value_mnk_49_128_128_k1_s1_oh7.h"

//1024*1024的256位对齐的char数组
// static char a[256*] __attribute__((aligned(256)));
// static char b[1024*1024] __attribute__((aligned(256)));
// static int  c[1024*1024] __attribute__((aligned(256)));

// val TaskTypeTensorLoad = 2.U(TypeBitWidth.W)
// val TaskTypeTensorZeroLoad = 3.U(TypeBitWidth.W) //直接将数据填充为0，实际上一直写SRAM，全0的值
// val TaskTypeTensorRepeatRowLoad = 4.U(TypeBitWidth.W) //重复加载一行数据，实际上是只读取了一行，然后一直反复写SRAM

//                                                   VVVVV 每个ReduceDim需要Load的周期数 VVVVV = 6*6 + 6*7 + 6*6 + 6*7 + 7*7 + 7*6 + 6*6 + 6*7 + 6*6
//AML的总load请求数量 = (9个kernel各自的load请求数量) * Tensor_K * Taskctrl拆分的任务次数 =  (361) * 2 * 64 = 46208[A_LOAD]
//                                                  ^^^^ = 2   ^^^^ = 49 * 512 * 512 --> 1 x 8 x 8 = 64
//                                                                     1    8     8
//                                                                 Tile_M Tile_N Tile_K

//BML的总load请求数量 = Tensor_N * (9个kernel) * Tensor_K * Taskctrl拆分的任务次数 =  64 * 9 * 2 * 64 = 73728[B_LOAD]
//                       64          9             2         64

//CML的总load请求数量(Repeat row load) = (Tensor_N * 4 / 32) * 暂存区任务 = 64 * 4 / 32 * 8 = 64 [C_LOAD]
//                                        64       ^    ^     ^^^^ = Tile_M * Tile_N = 1 * 8 = 8
//                          存回去的数据格式为4byte---|    |----LLC的读数带宽是32byte/cycle

//CML的总store请求数量 = (APPLICATION_M * Tensor_N * 4 / 32) * 暂存区任务 = 49 * 64 * 4 / 32 * 8 = 3136 [D_STORE]

int main(void) {


    uint64_t res1 = 1;
    uint64_t A = input;
    uint64_t A_Stride = APPLICATION_K * sizeof(input[0][0]);
    uint64_t B = weight;
    uint64_t B_Stride = APPLICATION_K * sizeof(weight[0][0]);
    uint64_t C = bias;
    uint64_t C_Stride = APPLICATION_N * sizeof(bias[0]);
    uint64_t D = output;
    uint64_t D_Stride = APPLICATION_N * sizeof(output[0][0]);
    uint64_t element_type = CUTEDataTypeI8I8I32;//1byte per input
    uint64_t bias_type = TaskTypeTensorRepeatRowLoad;
    // uint64_t transpose_result = 0;
    uint64_t current_M_index = 0;
    uint64_t issue_val = issue_cute_conv_marco_inst(A, A_Stride, B, B_Stride, C, C_Stride, D, D_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K,KERNEL_STRIDE, element_type, bias_type, TRANSPOSE_RESULT, CONV_STRIDE,CONV_OH_MAX,CONV_OW_MAX,KERNEL_SIZE,CONV_OH_PER_ADD,CONV_OW_PER_ADD,CONV_OH_INDEX,CONV_OW_INDEX);

    printf("issue_val: %ld\n", issue_val);
    //查询指令FIFO的情况
    res1 = cute_marco_inst_fifo_valid_search();
    if(res1){
        printf("FIFO not empty\n");
    }else{
        printf("FIFO empty\n");
        return -1;
    }

    printf("D Test start\n");
    res1 = cute_marco_inst_fifo_finish_search();
    while(!res1)
    {
        printf("Waiting for finish\n");
        res1 = cute_marco_inst_fifo_finish_search();
    }

    printf("finish\n");
    YGJK_INS_RRR(res1, 0, 0, 2);
	printf("acc time: %ldcycles\n", res1);
    YGJK_INS_RRR(res1, 0, 0, 5);
    printf("compute: %ldcycles\n", res1);
	YGJK_INS_RRR(res1, 0, 0, 3);
	printf("acc read req: %ld\n", res1);
	YGJK_INS_RRR(res1, 0, 0, 4);
	printf("acc write req: %ld\n", res1);

    

  return 0;
}
