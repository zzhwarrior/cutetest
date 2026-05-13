#include "ygjk.h"
#include <stdint.h>
//funct === 0，将配置好的Marco指令加入指令FIFO

//funct === 1    配置加速器，cfgData1 = ATensor的起始地址，cfgData2 = next_reduce_dim的stride
//funct === 2    配置加速器，cfgData1 = BTensor的起始地址，cfgData2 = next_reduce_dim的stride
//funct === 3    配置加速器，cfgData1 = CTensor的起始地址，cfgData2 = next_reduce_dim的stride
//funct === 4    配置加速器，cfgData1 = DTensor的起始地址，cfgData2 = next_reduce_dim的stride

//funct === 5    配置加速器，cfgData1 = (M[0~19bit]，N[20~39bit]，K[40~59bit])
//               对于卷积就是cfgData1 = (ohow[0~19bit]，oc[20~39bit]，ic[40~59bit])
//                         cfgData2 = kernel_stride 对于矩阵乘来说是0，对于卷积来说是下一个卷积核的起始地址卷积核是(oc,kh,kw,ic)排的

//funct === 6    配置加速器，cfgData1 = (element_type[0~7bit]，bias_type[8~15bit]，transpose_result[16~23bit],conv_stride[24~31bit],conv_oh_max[32~47bit],conv_ow_max[48~63bit])
//               配置加速器，cfgData2 = (kernel_size[0~7bit]，conv_oh_per_add[16~25]，conv_ow_per_add[26~35]， conv_oh_index[36~45bit],conv_oh_index[46~55bit])
//val conv_oh_per_add //避免在计算过程中进行除法运算，这里可以提前计算好
//val conv_ow_per_add //避免在计算过程中进行取余运算，这里可以提前计算好
#define CUTEDataTypeI8I8I32     0     //I8 * I8 * I32
#define CUTEDataTypeF16F16F32   1     //FP16 * FP16 * FP32
#define CUTEDataTypeBF16BF16F32 2     //BF16 * BF16 * FP32
#define CUTEDataTypeTF32TF32F32 3     //TF32 * TF32 * FP32
#define CUTEDataTypeI8U8I32     4     //I8 * UI8 * I32
#define CUTEDataTypeU8I8I32     5     //U8 * I8 * I32
#define CUTEDataTypeU8U8I32     6     //U8 * U8 * I32
#define CUTEDataTypee4m3F32     7


// int issue_Matmul_Marco_Inst()
#define TaskTypeTensorLoad  3
#define TaskTypeTensorZeroLoad  1
#define TaskTypeTensorRepeatRowLoad  2


#define Tensor_M_Element_Length 64
#define Tensor_N_Element_Length 64
#define Tensor_K_Element_Length 64


#define CUTE_CONFIG_FUNCTOPS 64

#define CUTE_ISSUE_MARCO_INST (CUTE_CONFIG_FUNCTOPS + 0)

#define CUTE_ATENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 1)
#define CUTE_BTENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 2)
#define CUTE_CTENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 3)
#define CUTE_DTENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 4)

#define CUTE_MNK_KERNALSTRIDE_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 5)
#define CUTE_CONV_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 6)
#define CUTE_FIFO_DEQUEUE_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 16)
#define CUTE_FIFO_GET_FINISH_TAIL_FIFOINDEX_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 17)

#define CUTE_SEARCH_FUNCTOPS 0
#define CUTE_IS_RUNNING_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 1)
#define CUTE_RUNNING_CYCLYES_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 2)
#define CUTE_MRMORY_LOAD_REQUEST_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 3)
#define CUTE_MRMORY_STORE_REQUEST_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 4)
#define CUTE_COMPUTE_CYCLYES_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 5)

#define CUTE_FIFO_FINISH_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 6)
#define CUTE_FIFO_FULL_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 7)
#define CUTE_FIFO_VALID_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 8)

uint64_t mrdcycle()
{
    uint64_t res1=0;
    asm volatile("rdcycle %0":"=r"(res1));
    return res1;
}

void issue_cute_config_ATensor(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride)
{
    int result;
    YGJK_INS_RRR(result, ATensor_Base_Addr, ATensor_M_Stride, CUTE_ATENSOR_CONFIG_FUNCTOPS);
}
void issue_cute_config_BTensor(uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride)
{
    int result;
    YGJK_INS_RRR(result, BTensor_Base_Addr, BTensor_M_Stride, CUTE_BTENSOR_CONFIG_FUNCTOPS);
}
void issue_cute_config_CTensor(uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride)
{
    int result;
    YGJK_INS_RRR(result, CTensor_Base_Addr, CTensor_M_Stride, CUTE_CTENSOR_CONFIG_FUNCTOPS);
}
void issue_cute_config_DTensor(uint64_t DTensor_Base_Addr,uint64_t DTensor_M_Stride)
{
    int result;
    YGJK_INS_RRR(result, DTensor_Base_Addr, DTensor_M_Stride, CUTE_DTENSOR_CONFIG_FUNCTOPS);
}

//数值范围检测M N K,16384
void issue_cute_config_MNK_KERNALSTRIDE(uint64_t M,uint64_t N,uint64_t K,uint64_t kernel_stride)
{
    int t;
    M = M & 0xFFFF;
    N = N & 0xFFFF;
    K = K & 0xFFFF;
    uint64_t cfgData1 = M | (N << 20) | (K << 40);
    YGJK_INS_RRR(t, cfgData1, kernel_stride, CUTE_MNK_KERNALSTRIDE_CONFIG_FUNCTOPS);
}

void issue_cute_config_CONV(uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t conv_stride,uint64_t conv_oh_max,uint64_t conv_ow_max,
                                    uint64_t kernel_size,uint64_t conv_oh_per_add,uint64_t conv_ow_per_add,uint64_t conv_oh_index,uint64_t conv_ow_index)
{
    int t;
    element_type = element_type & 0xFF;
    bias_type = bias_type & 0xFF;
    transpose_result = transpose_result & 0xFF;
    conv_stride = conv_stride & 0xFF;
    conv_oh_max = conv_oh_max & 0x7FFF;
    conv_ow_max = conv_ow_max & 0x7FFF;
    kernel_size = kernel_size & 0xF;
    conv_oh_per_add = conv_oh_per_add & 0x7FFF;
    conv_ow_per_add = conv_ow_per_add & 0x7FFF;
    conv_oh_index = conv_oh_index & 0x7FFF;
    conv_ow_index = conv_ow_index & 0x7FFF;
    uint64_t cfgData1 = element_type | (bias_type << 8) | (transpose_result << 16) | (conv_stride << 24) | (conv_oh_max << 32) | (conv_ow_max << 48);
    uint64_t cfgData2 = kernel_size  | (conv_oh_per_add << 4) | (conv_ow_per_add << 19) | (conv_oh_index << 34) | (conv_ow_index << 49);
    YGJK_INS_RRR(t, cfgData1, cfgData2, CUTE_CONV_CONFIG_FUNCTOPS);
}

void issue_cute_config_MatMul(uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t current_M_index)
{
    int t;
    element_type = element_type & 0xFF;
    bias_type = bias_type & 0xFF;
    transpose_result = transpose_result & 0xFF;
    uint64_t conv_stride = 1;
    uint64_t conv_oh_max = 0;
    uint64_t conv_ow_max = 0;//默认最大N
    uint64_t kernel_size = 1;
    uint64_t conv_oh_per_add = 0;
    uint64_t conv_ow_per_add = Tensor_M_Element_Length;//
    uint64_t conv_oh_index = 0;
    uint64_t conv_ow_index = current_M_index;
    uint64_t cfgData1 = element_type | (bias_type << 8) | (transpose_result << 16) | (conv_stride << 24) | (conv_oh_max << 32) | (conv_ow_max << 48);
    uint64_t cfgData2 = kernel_size  | (conv_oh_per_add << 4) | (conv_ow_per_add << 19) | (conv_oh_index << 34) | (conv_ow_index << 49);
    YGJK_INS_RRR(t, cfgData1, cfgData2, CUTE_CONV_CONFIG_FUNCTOPS);
}

uint64_t issue_cute_marco_inst()
{
    uint64_t res1=1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_ISSUE_MARCO_INST);
    return res1;
}

uint64_t issue_cute_conv_marco_inst(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride,
                                       uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride,
                                       uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride,
                                       uint64_t DTensor_Base_Addr,uint64_t DTensor_M_Stride,
                                       uint64_t M,uint64_t N,uint64_t K,uint64_t kernel_stride,
                                       uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t conv_stride,uint64_t conv_oh_max,uint64_t conv_ow_max,
                                       uint64_t kernel_size,uint64_t conv_oh_per_add,uint64_t conv_ow_per_add,uint64_t conv_oh_index,uint64_t conv_ow_index)
{
    issue_cute_config_ATensor(ATensor_Base_Addr,ATensor_M_Stride);
    issue_cute_config_BTensor(BTensor_Base_Addr,BTensor_M_Stride);
    issue_cute_config_CTensor(CTensor_Base_Addr,CTensor_M_Stride);
    issue_cute_config_DTensor(DTensor_Base_Addr,DTensor_M_Stride);
    issue_cute_config_MNK_KERNALSTRIDE(M,N,K,kernel_stride);
    issue_cute_config_CONV(element_type,bias_type,transpose_result,conv_stride,conv_oh_max,conv_ow_max,
                           kernel_size,conv_oh_per_add,conv_ow_per_add,conv_oh_index,conv_ow_index);
    return issue_cute_marco_inst();
}

uint64_t  issue_cute_matmul_marco_inst(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride,
                                       uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride,
                                       uint64_t BiasTensor_Base_Addr,uint64_t BiasTensor_M_Stride,
                                       uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride,
                                       uint64_t M,uint64_t N,uint64_t K,
                                       uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t matmul_m_index)
{
    issue_cute_config_ATensor(ATensor_Base_Addr,ATensor_M_Stride);
    issue_cute_config_BTensor(BTensor_Base_Addr,BTensor_M_Stride);
    issue_cute_config_CTensor(BiasTensor_Base_Addr,BiasTensor_M_Stride);
    issue_cute_config_DTensor(CTensor_Base_Addr,CTensor_M_Stride);
    issue_cute_config_MNK_KERNALSTRIDE(M,N,K,0);
    issue_cute_config_MatMul(element_type,bias_type,transpose_result,matmul_m_index);
    return issue_cute_marco_inst();
}

uint64_t cute_marco_inst_fifo_valid_search()
{
    uint64_t res1=1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_VALID_SEARCH_FUNCTOPS);
    return res1;
}

uint64_t cute_marco_inst_fifo_full_search()
{
    uint64_t res1=1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_FULL_SEARCH_FUNCTOPS);
    return res1;
}

uint64_t cute_marco_inst_fifo_finish_search()
{
    uint64_t res1=1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_FINISH_SEARCH_FUNCTOPS);
    return res1;
}

uint64_t cute_marco_inst_fifo_dequeue()
{
    uint64_t res1=1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_DEQUEUE_FUNCTOPS);
    return res1;
}

uint64_t cute_marco_inst_fifo_get_finish_tail_fifoindex()
{
    uint64_t res1=1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_GET_FINISH_TAIL_FIFOINDEX_FUNCTOPS);
    return res1;
}

uint64_t cute_marco_inst_tma_test()
{
    uint64_t res1=1;
    YGJK_INS_CUSTOM1_RRR(res1, 0, 0, 0);
    return res1;
}