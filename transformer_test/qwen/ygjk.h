// this part is used in RISC_V
#define CUSTOM0 0x0B
#define CUSTOM1 0x2B 
#define CUSTOM2 0x5B
#define CUSTOM3 0x7B

#define GET_VALUE1(x) #x
#define GET_VALUE(x) GET_VALUE1(x) 

#define YGJK_BIT(opcode,rd,xs2,xs1,xd,rs1,rs2,funct7)  \
    opcode                      | \
    (rd     <<(7))              | \
    (xs2    <<(7+5))            | \
    (xs1    <<(7+5+1))          | \
    (xd     <<(7+5+1+1))        | \
    (rs1    <<(7+5+1+1+1))      | \
    (rs2    <<(7+5+1+1+1+5))    | \
    (funct7 <<(7+5+1+1+1+5+5))

/*#define YGJK_INS_RRR(rd, rs1, rs2,fun)                      \ 
do{                                                           \
    register volatile long ins_c __asm__("t0");             \
    register volatile long ins_a __asm__("t1"); \
    ins_a = rs1; \
    register volatile long ins_b __asm__("t2") = (long)rs2; \
    __asm__ __volatile__ (".word " GET_VALUE(YGJK_BIT(CUSTOM0,5,1,1,1,6,7,fun)) "\n\t");              \
    rd = ins_a;                                             \
}while(0)*/

#define YGJK_CTRL_FUNC(fun)                      \
{                                                           \
    __asm__ __volatile__ (                                  \
        ".word " GET_VALUE(YGJK_BIT(CUSTOM1,5,0,0,0,6,7,fun)) "\n\t"              \
        );                                                  \
}

// TODO bug to be fixed
#define YGJK_TLBW(rs1, rs2)                                 \ 
{                                                           \
    __asm__ __volatile__ (                                  \
        "sd t1, -16(sp)\n\t"                                    \        
        "sd t2,  -8(sp)\n\t"                                    \        
        "add t1, zero, %0\n\t"                                  \        
        "add t2, zero, %1\n\t"                                  \        
        ".word " GET_VALUE(YGJK_BIT(CUSTOM1,5,0,1,1,6,7,4)) "\n\t"              \
        "ld t1, -16(sp)\n\t"                                    \        
        "ld t2,  -8(sp)\n\t"                                    \        
        :                                                   \        
        :"r" (rs1) , "r" (rs2)                              \        
        );                                                  \
}


// TODO bug to be fixed
#define YGJK_TEST(rd, rs1, rs2,fun)                      \ 
{                                                           \
    __asm__ __volatile__ (                                  \
        "sd t0, -24(sp)\n\t"                                \    
        "sd t1, -16(sp)\n\t"                                    \
        "sd t2,  -8(sp)\n\t"                                    \
        "add t1, zero, %1\n\t"                                  \   
        "add t2, zero, %2\n\t"                                  \
        ".word " GET_VALUE(YGJK_BIT(CUSTOM1,5,1,1,1,6,7,fun)) "\n\t"              \
        "add %0, zero, t0\n\t"                                  \
        "ld t0, -24(sp)\n\t"                                    \
        "ld t1, -16(sp)\n\t"                                    \
        "ld t2,  -8(sp)\n\t"                                    \
        :"=r"(rd)                                           \
        :"r" (rs1) , "r" (rs2)                              \
        );                                                  \
}

#define YGJK_INS_RRR(rd, rs1, rs2,fun)                      \ 
{                                                           \
    __asm__ __volatile__ (                                  \
        "sd t0, -24(sp)\n\t"                                \
        "sd t1, -16(sp)\n\t"                                    \
        "sd t2,  -8(sp)\n\t"                                    \
        "add t1, zero, %1\n\t"                                  \
        "add t2, zero, %2\n\t"                                  \
        ".word " GET_VALUE(YGJK_BIT(CUSTOM0,5,1,1,1,6,7,fun)) "\n\t"              \
        "add %0, zero, t0\n\t"                                  \
        "ld t0, -24(sp)\n\t"                                    \
        "ld t1, -16(sp)\n\t"                                    \
        "ld t2,  -8(sp)\n\t"                                    \
        :"=r"(rd)                                           \
        :"r" (rs1) , "r" (rs2)                              \
        :"t0","t1","t2","memory"                                     \
        );                                                  \
}

#define YGJK_INS_CUSTOM1_RRR(rd, rs1, rs2,fun)                      \ 
{                                                           \
    __asm__ __volatile__ (                                  \
        "sd t0, -24(sp)\n\t"                                \
        "sd t1, -16(sp)\n\t"                                    \
        "sd t2,  -8(sp)\n\t"                                    \
        "add t1, zero, %1\n\t"                                  \
        "add t2, zero, %2\n\t"                                  \
        ".word " GET_VALUE(YGJK_BIT(CUSTOM1,5,1,1,1,6,7,fun)) "\n\t"              \
        "add %0, zero, t0\n\t"                                  \
        "ld t0, -24(sp)\n\t"                                    \
        "ld t1, -16(sp)\n\t"                                    \
        "ld t2,  -8(sp)\n\t"                                    \
        :"=r"(rd)                                           \
        :"r" (rs1) , "r" (rs2)                              \
        :"t0","t1","t2","memory"                                     \
        );                                                  \
}

// UNUSED
#define YGJK_INS_XRR(rd, rs1, rs2,fun)                      \ 
{                                                           \
    __asm__ __volatile__ (                                  \
        "sd t1, -16(sp)\n\t"                                    \        
        "sd t2,  -8(sp)\n\t"                                    \        
        "add t1, zero, %0\n\t"                                  \        
        "add t2, zero, %1\n\t"                                  \        
        ".word " GET_VALUE(YGJK_BIT(CUSTOM0,5,1,1,0,6,7,fun)) "\n\t"              \
        "ld t1, -16(sp)\n\t"                                    \        
        "ld t2,  -8(sp)\n\t"                                    \        
        :                                                      \        
        :"r" (rs1) , "r" (rs2)                              \        
        );                                                  \
}
