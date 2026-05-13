#include <stdio.h>
// #include <riscv-pk/encoding.h>
// #include <riscv-pk/marchid.h>
#include "marchid.h"
#include <stdint.h>
#include "cuteMarcoinstHelper.h"
#include "conv_value_mnk_49_128_128_k1_s1_oh7.h"

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
    uint64_t element_type = 1;//1byte per input
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
