/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __HCCL_TESTER_H__
#define __HCCL_TESTER_H__

typedef enum tag_hccl_excute_type /* 动作类型枚举 */
{ EXCUTE_TYPE_BROADCAST = 0,
  EXCUTE_TYPE_REDUCE = 1,
  EXCUTE_TYPE_ALL_GATHER = 2,
  EXCUTE_TYPE_REDUCE_SCATTER = 3,
  EXCUTE_TYPE_ALL_REDUCE = 4,
  EXCUTE_TYPE_SEND_RECEIVE = 5,
  EXCUTE_TYPE_RESERVED,
} hccl_excute_t;

typedef enum tag_hccl_sendbuf_init_type {
    HCCL_SENDBUF_INIT_TYPE_INC = 0,    /* 初始值全局递增 */
    HCCL_SENDBUF_INIT_TYPE_ALL0 = 1,   /* 初始值全0 */
    HCCL_SENDBUF_INIT_TYPE_ALL1 = 2,   /* 初始值全1 */
    HCCL_SENDBUF_INIT_TYPE_INC_S8 = 3, /* 在s8数据类型表示的范围内递增 */
    HCCL_SENDBUF_INIT_TYPE_DEVID = 4,  /* 初始值为rankid */
    HCCL_SENDBUF_INIT_TYPE_OFFSET = 5, /* 初始值为数据在buffer内的偏移 */
    HCCL_SENDBUF_INIT_TYPE_RESERVED,

} hccl_sendbuf_init_type_t;

typedef enum tag_comms_init_type {
    COMMS_INIT_TYPE_ALL = 0,     /* 调用hcclCommInitAll，只能进程内 */
    COMMS_INIT_TYPE_BY_RANK = 1, /* 调用hcclCommInitRank，可以跨进程 */
    COMMS_INIT_TYPE_RESERVED,
} comms_init_type_t;

typedef enum tag_task_run_type {
    TASK_RUN_SERIAL = 0,   /* 以rank为序下发，可以针对保序下发 */
    TASK_RUN_PARALLEL = 1, /* 以操作为序下发 */
    TASK_RUN_RESERVED
} task_run_type_t;

// 控制打印信息的掩码
#define PRINT_MASK_SEND 0x1   /* 打印发送sendbuf */
#define PRINT_MASK_RECV 0x10  /* 打印接收recvbuf */
#define PRINT_MASK_RSLT 0x100 /* 打印预期结果resultbuf */

#define INVALID_RANK -1 /* 无效的rank_id */

#define SEND_RECV_VALID 0xff
#define FLOAT_MAX_DIFF_RANGE 0.01f /* 浮点数最大误差范围 */

typedef struct send_receive_struct /* send receive rank信息结构体 */
{
    s32 src_valid;
    s32 dest_valid;
    s32 src_rank;
    s32 dest_rank;
    s32 tag;
} send_receive_t;

typedef struct excute_para_struct {
    void* sendbuff;
    void* recvbuff;
    s32 count;
    HcclDataType datatype;
    HcclComm comm;
    rtStream_t stream;
    s32 stream_id;
    HcclReduceOp op;
    s32 root;
    hccl_excute_t op_type;
    send_receive_t src_dest_info;
    s32 dev_id;
    s32 task_num;
} excute_para_t;

typedef struct test_task /* 测试任务结构体，每个对象是一个测试任务 */
{
    hccl_excute_t excute_type;
    std::vector<void*> dev_sendbuf;
    std::vector<void*> dev_recvbuf;
    std::vector<void*> host_sendbuf;
    std::vector<void*> host_recvbuf;
    void* result_buf;
    std::vector<sal_thread_t> tids;
    HcclDataType data_type;
    HcclReduceOp reduce_op;
    s32 count;
    hccl_sendbuf_init_type_t init_type;
    s32 buf_print_enable;
    s32 root;
    s32 send_rank;
    s32 recv_rank;
    s32 sendbuf_num;
    s32 recvbuf_num;
    s32 rsltbuf_num;

} test_task_t;

/* bcast */
typedef struct bcast_excute_para {
    s32 root;
} bcast_excute_para_t;

/* reduce */
typedef struct reduce_excute_para {
    s32 root;
    HcclReduceOp reduce_op;
} reduce_excute_para_t;

/* reduce_scatter */
typedef struct reduce_scatter_excute_para {
    HcclReduceOp reduce_op;
} reduce_scatter_excute_para_t;

/* all_reduce */
typedef struct allreduce_excute_para {
    HcclReduceOp reduce_op;
} allreduce_excute_para_t;

/* send recv操作 */
typedef struct send_recv_excute_para {
    s32 send_rank;
    s32 recv_rank;
} send_recv_excute_para_t;

typedef struct test_para /* 测试参数结构体 */
{
    HcclDataType data_type;
    s32 count;
    union {
        /* all_gather没有root和reduce操作 */
        bcast_excute_para bcast_para;
        reduce_excute_para reduce_para;
        reduce_scatter_excute_para rs_para;
        allreduce_excute_para allre_para;
        send_recv_excute_para p2p_para;
    } excute_para;

    hccl_sendbuf_init_type_t init_type;
    s32 buf_print_enable;
} test_para_t;

typedef struct tester_rank_info /* rank相关信息结构体 */
{
    s32 rank;
    s32 devid;
    HcclComm comm;
    s32 nranks;
    HcclRootInfo commId;
    rtStream_t stream;
    sal_thread_t tids;
} rank_info_t;

/* 初始化comminicator需要的参数信息 */
typedef struct comm_init_para_t {
    comms_init_type_t init_type; /* 初始化communicator的方式 */

    union {
        struct all_rank {
            s32* device_list; /* init all使用的device_id列表，对应COMMS_INIT_TYPE_ALL */
            s32 ndev;
        } all;

        struct unique_rank /* init rank只需要单个rank的信息，对应COMMS_INIT_TYPE_BY_RANK */
        {
            s32 my_rank;
            s32 dev_id;
            s32 nranks;
            HcclRootInfo* commId;
        } unique;
    } rank;

} comm_init_para;

typedef struct dev_info {
    s32 rank_id;
    s32 dev_id;
} dev_info_t;

class tester {
private:
    std::vector<tester_rank_info> rank_list; // rank信息的列表
    std::vector<test_task> task_vec;         // 测试任务列表
    s32 device_count;                        // device计数
    comms_init_type_t comm_init_type;        // comm的初始化方式，是否跨进程
    task_run_type_t run_type;                // 运行类型，线程或者进程

public:
    tester();

    /* 初始化comms, 只调用一次。当前仅支持节点内的CommInitAll */
    HcclResult init_comm(comm_init_para_t& init_para);

    /* 添加测试任务，可以多次调用，任务会添加到队列，一起执行 */
    HcclResult add_test_task(hccl_excute_t excute_type, test_para* para); // 改成list

    /* 执行测试任务，支持每次任务创建新线程或者所有任务用相同线程执行 */
    HcclResult run(task_run_type_t run_type);

    /* 等待运行结束后，检查结果 */
    HcclResult check_result(); // 检查完清空

    ~tester();

protected:
    /* 初始化所有comm，用于节点内集合通信 */
    HcclResult init_comm_all();

    /* 按rank执行comm，用于节点间集合通信 */
    HcclResult init_comm_by_rank();

    /* 分配测试任务所需的设备和host内存 */
    HcclResult excute_malloc_task_mem(test_task_t& task);

    /* 释放任务使用的设备和host内存 */
    HcclResult excute_free_task_mem(test_task_t& task);

    /* 检查add_test_task函数入参 */
    HcclResult check_task_para(hccl_excute_t excute_type, test_para* para);

    /* 根据excute类型计算所需的buf个数 */
    HcclResult get_buff_count(hccl_excute_t type, s32 count, s32* sendbuf_num, s32* recvbuf_num, s32* rsltbuf_num);

    /* 通过datatype获取该类型的长度 */
    HcclResult get_len_by_datatype(HcclDataType datatype, s32* len);

    /* 所有任务使用同一组线程执行 */
    HcclResult run_by_thread_fix();

    /* 每一个测试任务新建一组线程执行 */
    HcclResult run_by_thread();

    /* 新建进程执行任务，用于节点间集合通信 */
    HcclResult run_by_process();

    /* 等待所有任务执行完 */
    HcclResult wait_all_task_finish();

    /* 计算预期结果 */
    template <typename T>
    HcclResult compute_result(test_task_t* task, T type_para);

    /* 打印buf的信息 */
    template <typename T>
    HcclResult print_buf_info(test_task_t* task, T type_para);

    /* 根据指定的类型，给sendbuf赋值 */
    template <typename T>
    HcclResult init_sendbuf_value(test_task_t* task, T cpu_type);

    /* 比较集合通信的输出结果和预期结果 */
    template <typename T>
    HcclResult compare_result(test_task_t* task, T cpu_type);

    /* 检查一个任务的结果，包括计算预期结果，比较结果，打印信息，释放空间 */
    template <typename T>
    HcclResult check_single_task(test_task_t* task, T cpu_type);

    /* 申请二维内存 */
    template <typename T>
    HcclResult malloc_two_dimension_memory(T**& out_mem, s32 first_dimen, s32 second_dimen, T unit);

    /* 释放二维内存 */
    template <typename T>
    HcclResult free_two_dimension_memory(T** out_mem, s32 first_dimen);

    /* 将device buf同步到host buf, 用于预期结果计算和比较 */
    HcclResult dev_to_host_mem_synchronize();

    HcclResult destroy_comms();
};

#endif // __HCCL_TESTER_H__
