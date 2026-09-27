/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <pthread.h>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "../llt/hccl/stub/llt_hccl_stub_pub.h"
#include "tester_pub.h"

#define TEST_CHECK_RET_VALUE(cmd, value)    \
    do {                                    \
        HcclResult __ret__ = cmd;           \
        EXPECT_EQ(__ret__, HCCL_SUCCESS);   \
        if (__ret__ != HCCL_SUCCESS) {      \
            STUB_ERROR("ret[%d]", __ret__); \
            return value;                   \
        }                                   \
    } while (0)

/* 任务执行的入口函数 */
void* excute_task(void* parg)
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    excute_para_t* para = (excute_para_t*)parg;
    s32 ret;
    /* 第一个任务的task_num表示总任务数量 */
    s32 task_num = para[0].task_num;

    rtError_t rt_ret = aclrtSetDevice(para->dev_id);

    if (rt_ret != RT_ERROR_NONE) {
        STUB_ERROR("device[%d], rt_set_device fail ret[%d]", para->dev_id, rt_ret);
    }

    /* 入口函数可以执行多个任务 */
    for (int j = 0; j < task_num; j++) {
        hccl_excute_t op_type = para[j].op_type;

        switch (op_type) {
            case EXCUTE_TYPE_BROADCAST:
                break;

            case EXCUTE_TYPE_REDUCE:
                break;

            case EXCUTE_TYPE_ALL_GATHER:
                break;

            case EXCUTE_TYPE_REDUCE_SCATTER:
                break;

            case EXCUTE_TYPE_ALL_REDUCE:
                break;

            case EXCUTE_TYPE_SEND_RECEIVE:
                break;

            case EXCUTE_TYPE_RESERVED:
            default:
                break;
        }
    }

    return NULL;
}

tester::tester() { return; }

tester::~tester() { return; }

HcclResult tester::init_comm_all()
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    vector<tester_rank_info>::iterator rank_it;
    s32 loop = 0;

    /* comm和device_list是为调用hcclCommInitAll临时申请的空间，在相关信息保存到rank_list后释放 */
    HcclComm* comms = (HcclComm*)sal_malloc(sizeof(HcclComm) * rank_list.size());

    if (comms == NULL) {
        STUB_ERROR("init_comm malloc failed size[%d]", sizeof(HcclComm) * rank_list.size());
        return HCCL_E_MEMORY;
    }

    s32* device_list = (s32*)sal_malloc(sizeof(s32) * rank_list.size());

    if (device_list == NULL) {
        STUB_ERROR("init_comm malloc failed size[%d]", sizeof(s32) * rank_list.size());
        return HCCL_E_MEMORY;
    }

    for (rank_it = rank_list.begin(); rank_it != rank_list.end(); rank_it++) {
        device_list[loop++] = rank_it->devid;
    }

    //  通信域初始化
    hccl_ret = hcclCommInitAll(comms, device_count, &device_list[0]);
    EXPECT_EQ(hccl_ret, HCCL_SUCCESS);

    if (hccl_ret != HCCL_SUCCESS) {
        STUB_ERROR("hcclCommInitAll failed return value[%d], device_count[%d]", hccl_ret, device_count);
        return hccl_ret;
    }

    /* 创建stream */
    loop = 0;

    for (rank_it = rank_list.begin(); rank_it != rank_list.end(); rank_it++) {
        rank_it->comm = comms[loop++];

        rt_ret = aclrtSetDevice(rank_it->devid);
        EXPECT_EQ(rt_ret, ACL_SUCCESS);

        if (rt_ret != ACL_SUCCESS) {
            STUB_ERROR("device[%d] aclrtSetDevice fail return value[%d]", rank_it->devid, hccl_ret);
            return HCCL_E_INTERNAL;
        }

        rt_ret = aclrtCreateStream(&(rank_it->stream));
        EXPECT_EQ(rt_ret, ACL_SUCCESS);

        if (rt_ret != ACL_SUCCESS) {
            STUB_ERROR("device[%d] aclrtCreateStream fail return value[%d]", rank_it->devid, hccl_ret);
            return HCCL_E_INTERNAL;
        }
    }

    /* 释放临时内存 */
    sal_free(comms);
    sal_free(device_list);

    return HCCL_SUCCESS;
}

HcclResult tester::init_comm_by_rank() { return HCCL_E_NOT_SUPPORT; }

HcclResult tester::get_len_by_datatype(HcclDataType datatype, s32* len)
{
    switch (datatype) {
        case HCCL_DATA_TYPE_INT8:
            *len = 1;
            break;

        case HCCL_DATA_TYPE_FP16:
            *len = 2;
            break;

        case HCCL_DATA_TYPE_FP32:
            *len = 4;
            break;

            /* HCCL_DATA_TYPE_INT类型当前不支持 */
        default:
            STUB_ERROR("get_len_by_datatype fail datatype[%d]", datatype);
            return HCCL_E_NOT_SUPPORT;
    }

    return HCCL_SUCCESS;
}

HcclResult tester::get_buff_count(hccl_excute_t type, s32 count, s32* sendbuf_num, s32* recvbuf_num, s32* rsltbuf_num)
{
    HcclResult hccl_ret;

    switch (type) {
        case EXCUTE_TYPE_BROADCAST:
        case EXCUTE_TYPE_REDUCE:
        case EXCUTE_TYPE_ALL_REDUCE:
        case EXCUTE_TYPE_SEND_RECEIVE:
            *sendbuf_num = count;
            *recvbuf_num = count;
            *rsltbuf_num = count;
            break;

        case EXCUTE_TYPE_REDUCE_SCATTER:
            *sendbuf_num = count * device_count;
            *recvbuf_num = count;
            *rsltbuf_num = count * device_count;
            break;

        case EXCUTE_TYPE_ALL_GATHER:
            *sendbuf_num = count;
            *recvbuf_num = count * device_count;
            *rsltbuf_num = count * device_count;
            break;

            /* 申请最大量的buff */
        default:
            *sendbuf_num = count * device_count;
            *recvbuf_num = count * device_count;
            *rsltbuf_num = count * device_count;
            break;
    }

    return HCCL_SUCCESS;
}

HcclResult tester::init_comm(comm_init_para_t& init_para)
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    vector<s32>::iterator para_it;

    comm_init_type = init_para.init_type;

    switch (comm_init_type) {
        case COMMS_INIT_TYPE_ALL:

            /* 通过dev_list的大小，得到rank数量 */
            device_count = init_para.rank.all.ndev;

            /* 将rank相关信息保存下来，stream和comm需要另行创建 */
            for (s32 rank_id = 0; rank_id < device_count; rank_id++) {
                tester_rank_info rank;

                rank.rank = rank_id;
                rank.devid = init_para.rank.all.device_list[rank_id];
                rank.nranks = device_count;
                rank.stream = NULL;
                rank.comm = NULL;
                rank_list.push_back(rank);
            }

            return init_comm_all();

        case COMMS_INIT_TYPE_BY_RANK:
            tester_rank_info rank;

            rank.rank = init_para.rank.unique.my_rank;
            rank.devid = init_para.rank.unique.dev_id;
            rank.nranks = init_para.rank.unique.nranks;
            rank.stream = NULL;
            rank.comm = NULL;

            sal_memcpy(&rank.commId, sizeof(HcclRootInfo), init_para.rank.unique.commId, sizeof(HcclRootInfo));
            ;

            rank_list.push_back(rank);

            return init_comm_by_rank();

        default:
            return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult tester::check_task_para(hccl_excute_t excute_type, test_para* para)
{
    /* 集合通信类型检查 */
    if ((excute_type < EXCUTE_TYPE_BROADCAST) || (excute_type >= EXCUTE_TYPE_RESERVED)) {
        STUB_ERROR("add_test_task err para, excute_type[%d]", excute_type);
        return HCCL_E_PARA;
    }

    if (para == NULL) {
        STUB_ERROR("add_test_task err para, input para is null");
        return HCCL_E_PARA;
    }

    /* 数据类型检查 */
    if ((para->data_type < HCCL_DATA_TYPE_INT8) || (para->data_type >= HCCL_DATA_TYPE_RESERVED)) {
        STUB_ERROR("add_test_task err para, para->data_type[%d]", para->data_type);
        return HCCL_E_NOT_SUPPORT;
    }

    /* sendbuf赋值类型检查 */
    if ((para->init_type < HCCL_SENDBUF_INIT_TYPE_INC) || (para->init_type >= HCCL_SENDBUF_INIT_TYPE_RESERVED)) {
        STUB_ERROR("add_test_task err para, para->init_type[%d]", para->init_type);
        return HCCL_E_NOT_SUPPORT;
    }

    return HCCL_SUCCESS;
}

HcclResult tester::add_test_task(hccl_excute_t excute_type, test_para* para)
{
    HcclResult hccl_ret;

    /* 参数检查 */
    hccl_ret = check_task_para(excute_type, para);
    TEST_CHECK_RET_VALUE(hccl_ret, hccl_ret);

    test_task_t task;
    task.excute_type = excute_type;
    task.data_type = para->data_type;
    task.count = para->count;
    task.init_type = para->init_type;
    task.buf_print_enable = para->buf_print_enable;

    /* 集合通信不同类型特有的参数赋值 */
    switch (excute_type) {
        case EXCUTE_TYPE_BROADCAST:
            task.root = para->excute_para.bcast_para.root;
            task.reduce_op = HCCL_REDUCE_RESERVED;
            break;

        case EXCUTE_TYPE_REDUCE:
            task.root = para->excute_para.reduce_para.root;
            task.reduce_op = para->excute_para.reduce_para.reduce_op;
            break;

        case EXCUTE_TYPE_REDUCE_SCATTER:
            task.reduce_op = para->excute_para.rs_para.reduce_op;
            break;

        case EXCUTE_TYPE_ALL_REDUCE:
            task.reduce_op = para->excute_para.allre_para.reduce_op;
            break;

        case EXCUTE_TYPE_SEND_RECEIVE:
            task.send_rank = para->excute_para.p2p_para.send_rank;
            task.recv_rank = para->excute_para.p2p_para.recv_rank;
            task.reduce_op = HCCL_REDUCE_RESERVED;
            break;

        case EXCUTE_TYPE_ALL_GATHER:
            task.reduce_op = HCCL_REDUCE_RESERVED;

        default:
            break;
    }

    /* 申请任务内存 */
    hccl_ret = excute_malloc_task_mem(task);
    TEST_CHECK_RET_VALUE(hccl_ret, HCCL_E_INTERNAL);

    /* 为sendbuf赋初始值 */
    float cpu_tpye_float = 0.0f;
    s8 cpu_tpye_s8 = 0;

    if (task.data_type == HCCL_DATA_TYPE_FP32) {
        hccl_ret = init_sendbuf_value(&task, cpu_tpye_float);
    } else {
        hccl_ret = init_sendbuf_value(&task, cpu_tpye_s8);
    }

    /* 添加进入任务组 */
    task_vec.push_back(task);

    return hccl_ret;
}

HcclResult tester::excute_malloc_task_mem(test_task_t& task)
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    s32 length;
    void* new_buf = NULL;
    void* new_host_buf = NULL;

    /* 获取数据长度和所需buff容量 */
    hccl_ret = get_len_by_datatype(task.data_type, &length);
    TEST_CHECK_RET_VALUE(hccl_ret, HCCL_E_INTERNAL);

    hccl_ret
        = get_buff_count(task.excute_type, task.count, &(task.sendbuf_num), &(task.recvbuf_num), &(task.rsltbuf_num));
    TEST_CHECK_RET_VALUE(hccl_ret, HCCL_E_INTERNAL);

    /* 同时申请sendbuf和recvbuf的设备内存 */
    for (s32 i = 0; i < device_count; ++i) {
        rt_ret = aclrtSetDevice(rank_list[i].devid);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        if (rt_ret != RT_ERROR_NONE) {
            STUB_ERROR("device[%d] aclrtSetDevice fail return value[%d]", i, hccl_ret);
            break;
        }

        aclrtMallocAttrValue moduleIdValue;
        moduleIdValue.moduleId = HCCL;
        aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
        aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

        aclError aclRet = aclrtMallocWithCfg(&new_buf, task.sendbuf_num * length, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);

        if (aclRet != ACL_SUCCESS) {
            STUB_ERROR(
                "device[%d] aclrtMallocWithCfg sendbuf size[%d] fail return value[%d]", i, task.sendbuf_num * length,
                rt_ret);
            return HCCL_E_MEMORY;
        }

        /* 保存申请的sendbuf设备内存 */
        task.dev_sendbuf.push_back(new_buf);

        aclError aclRet = aclrtMallocWithCfg(&new_buf, task.recvbuf_num * length, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);

        if (aclRet != ACL_SUCCESS) {
            STUB_ERROR(
                "device[%d] aclrtMallocWithCfg recvbuf size[%d] fail return value[%d]", i, task.sendbuf_num * length,
                rt_ret);
            return HCCL_E_MEMORY;
        }

        /* 保存申请的recvbuf设备内存 */
        task.dev_recvbuf.push_back(new_buf);
    }

    /* 同时申请sendbuf和recvbuf的host内存 */
    for (s32 i = 0; i < device_count; ++i) {
        new_host_buf = (void*)sal_malloc(task.sendbuf_num * length);
        EXPECT_NE(new_host_buf, (void*)NULL);

        if (new_host_buf == NULL) {
            STUB_ERROR("malloc host sendbuf memory size(%d) failed", task.sendbuf_num * length);
            return HCCL_E_MEMORY;
        }

        task.host_sendbuf.push_back(new_host_buf);

        new_host_buf = (void*)sal_malloc(task.recvbuf_num * length);
        EXPECT_NE(new_host_buf, (void*)NULL);

        if (new_host_buf == NULL) {
            STUB_ERROR("malloc host recvbuf memory size(%d) failed", task.recvbuf_num * length);
            return HCCL_E_MEMORY;
        }

        task.host_recvbuf.push_back(new_host_buf);
    }

    /* 申请预期结果resultbuf使用的内存 */
    task.result_buf = (void*)sal_malloc(task.rsltbuf_num * length);
    EXPECT_NE(task.result_buf, (void*)NULL);

    if (task.result_buf == NULL) {
        STUB_ERROR("malloc host memory size(%d) failed", task.rsltbuf_num * length);
        return HCCL_E_MEMORY;
    }

    return HCCL_SUCCESS;
}

HcclResult tester::excute_free_task_mem(test_task_t& task)
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    /* 释放申请的sendbuf和recvbuf的设备内存 */
    for (s32 i = 0; i < device_count; i++) {
        rt_ret = aclrtSetDevice(rank_list[i].devid);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        if (rt_ret != RT_ERROR_NONE) {
            STUB_ERROR("device[%d] aclrtFree fail return value[%d]", i, rt_ret);
            hccl_ret = HCCL_E_INTERNAL;
            continue;
        }

        if (task.dev_sendbuf[i] != NULL) {
            rt_ret = aclrtFree(task.dev_sendbuf[i]);
            EXPECT_EQ(rt_ret, RT_ERROR_NONE);

            if (rt_ret != RT_ERROR_NONE) {
                STUB_ERROR("device[%d] aclrtFree sendbuf fail return value[%d]", i, rt_ret);
                hccl_ret = HCCL_E_INTERNAL;
            }
        }

        if (task.dev_recvbuf[i] != NULL) {
            rt_ret = aclrtFree(task.dev_recvbuf[i]);
            EXPECT_EQ(rt_ret, RT_ERROR_NONE);

            if (rt_ret != RT_ERROR_NONE) {
                STUB_ERROR("device[%d] aclrtFree recvbuf fail return value[%d]", i, rt_ret);
                hccl_ret = HCCL_E_INTERNAL;
            }
        }
    }

    /* 释放申请的sendbuf和recvbuf的host内存 */
    for (s32 i = 0; i < device_count; i++) {
        if (task.host_sendbuf[i] != NULL) {
            free(task.host_sendbuf[i]);
        }

        if (task.host_recvbuf[i] != NULL) {
            free(task.host_recvbuf[i]);
        }
    }

    /* 释放result_buf内存 */
    if (task.result_buf != NULL) {
        free(task.result_buf);
    }

    return hccl_ret;
}

template <typename T>
HcclResult tester::init_sendbuf_value(test_task_t* task, T cpu_type)
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    /* 需要初始赋值的数据个数 */
    s32 init_num = task->count;

    if (task->excute_type == EXCUTE_TYPE_REDUCE_SCATTER) {
        init_num = task->count * device_count;
    }

    for (s32 rank = 0; rank < device_count; rank++) {
        rt_ret = aclrtSetDevice(rank_list[rank].devid);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        if (rt_ret != RT_ERROR_NONE) {
            STUB_ERROR(
                "init_sendbuf_value set device failed,rank[%d], dev_id[%d], rt_ret[%d]", rank, rank_list[rank].devid,
                rt_ret);
            return HCCL_E_INTERNAL;
        }

        /* broadcast操作只给root节点赋值 */
        if (task->excute_type == EXCUTE_TYPE_BROADCAST && rank != task->root) {
            continue;
        }

        for (s32 offset = 0; offset < init_num; offset++) {
            /* 模板类型为float型 */
            if (task->data_type == HCCL_DATA_TYPE_FP32) {
                /* 按照sendbuf赋值的枚举类型为sendbuf赋出值 */
                switch (task->init_type) {
                    case HCCL_SENDBUF_INIT_TYPE_ALL0:
                        ((T*)task->host_sendbuf[rank])[offset] = 0.0f;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_ALL1:
                        ((T*)task->host_sendbuf[rank])[offset] = 1.0f;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_INC:
                        ((T*)task->host_sendbuf[rank])[offset] = (task->count * rank + offset) / 100.0f;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_DEVID:
                        ((T*)task->host_sendbuf[rank])[offset] = (float)rank / 1.0f;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_OFFSET:
                        ((T*)task->host_sendbuf[rank])[offset] = (float)offset / 1.0f;
                        break;

                    default:
                        ((T*)task->host_sendbuf[rank])[offset] = (task->count * rank + offset) / 100.0f;
                        break;
                }
            }
            /* 非浮点数据类型 */
            else {
                switch (task->init_type) {
                    case HCCL_SENDBUF_INIT_TYPE_ALL0:
                        ((T*)task->host_sendbuf[rank])[offset] = 0;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_ALL1:
                        ((T*)task->host_sendbuf[rank])[offset] = 1;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_INC_S8:
                        ((T*)task->host_sendbuf[rank])[offset] = offset % 15;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_INC:
                        ((T*)task->host_sendbuf[rank])[offset] = task->count * rank + offset;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_DEVID:
                        ((T*)task->host_sendbuf[rank])[offset] = rank;
                        break;

                    case HCCL_SENDBUF_INIT_TYPE_OFFSET:
                        ((T*)task->host_sendbuf[rank])[offset] = offset;
                        break;

                    default:
                        ((T*)task->host_sendbuf[rank])[offset] = task->count * rank + offset;
                        break;
                }
            }
        }

        /* 将host_sendbuf信息异步copy到设备内存中 */
        s64 temp = (s64)init_num * sizeof(T);
        rt_ret = aclrtMemcpyAsync(
            task->dev_sendbuf[rank], (u64)temp, task->host_sendbuf[rank], (u64)temp, ACL_MEMCPY_HOST_TO_DEVICE,
            rank_list[rank].stream);

        if (rt_ret != ACL_SUCCESS) {
            STUB_ERROR(
                "init_sendbuf_value rt_memcpy_async failed, rank[%d], dev_id[%d], rt_ret[%d]", rank,
                rank_list[rank].devid, rt_ret);
            return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult tester::run_by_thread_fix(void)
{
    HcclResult hccl_ret = HCCL_SUCCESS;

    /* 取出测试任务队列中的excute，顺序执行 */
    vector<test_task>::iterator task_it;

    for (task_it = task_vec.begin(); task_it != task_vec.end(); task_it++) {
        ;
    }

    return hccl_ret;
}

template <typename T>
HcclResult tester::malloc_two_dimension_memory(T**& out_mem, s32 first_dimen, s32 second_dimen, T unit)
{
    /* 入参判断 */

    /* 申请第一维 */
    out_mem = (T**)malloc(sizeof(T*) * first_dimen);

    if (out_mem == NULL) {
        STUB_ERROR("dimension1 malloc size[%d] failed", first_dimen);
        return HCCL_E_MEMORY;
    }

    /* 申请第二维 */
    for (s32 i = 0; i < first_dimen; i++) {
        out_mem[i] = (T*)malloc(second_dimen * sizeof(unit));

        if (out_mem[i] == NULL) {
            STUB_ERROR("dimension2[%d] malloc size[%d] failed", i, second_dimen * sizeof(unit));
            return HCCL_E_MEMORY;
        }
    }

    return HCCL_SUCCESS;
}

template <typename T>
HcclResult tester::free_two_dimension_memory(T** out_mem, s32 first_dimen)
{
    if (out_mem == NULL) {
        return HCCL_E_MEMORY;
    }

    /* 释放第二维度内存 */
    for (s32 i = 0; i < first_dimen; i++) {
        if (out_mem[i] != NULL) {
            free(out_mem[i]);
        } else {
            /* 如果有二维内存为空，后续的内存不会继续申请，跳出释放 */
            break;
        }
    }

    free(out_mem);

    return HCCL_SUCCESS;
}

HcclResult tester::run_by_thread(void)
{
    rtError_t rt_ret = RT_ERROR_NONE;
    HcclResult hccl_ret = HCCL_SUCCESS;
    sal_thread_t temp_tid;
    s32 sret = 0;

    excute_para_t** para_info;
    excute_para_t para_unit;

    hccl_ret = malloc_two_dimension_memory(para_info, device_count, task_vec.size(), para_unit);

    if (hccl_ret != HCCL_SUCCESS) {
        STUB_ERROR("run_by_thread malloc_two_dimension_memory failed size: [%d] * [%d]", device_count, task_vec.size());
        return HCCL_E_MEMORY;
    }

    /* 取出测试任务队列中的excute，顺序执行 */
    vector<test_task>::iterator task_it;
    s32 j = 0;

    for (task_it = task_vec.begin(); task_it != task_vec.end(); task_it++) {
        for (s32 i = 0; i < device_count; i++) {
            para_info[i][j].op_type = task_it->excute_type;
            para_info[i][j].comm = rank_list[i].comm;
            para_info[i][j].count = task_it->count;
            para_info[i][j].datatype = task_it->data_type;
            para_info[i][j].sendbuff = task_it->dev_sendbuf[i];
            para_info[i][j].recvbuff = task_it->dev_recvbuf[i];
            para_info[i][j].stream = rank_list[i].stream;
            para_info[i][j].dev_id = rank_list[i].devid;
            para_info[i][j].root = task_it->root;

            /* 涉及reduce操作的类型 */
            if (task_it->excute_type == EXCUTE_TYPE_REDUCE || task_it->excute_type == EXCUTE_TYPE_ALL_REDUCE
                || task_it->excute_type == EXCUTE_TYPE_REDUCE_SCATTER) {
                para_info[i][j].op = task_it->reduce_op;
            }
        }

        /* send receive操作需要的数据 */
        if (task_it->excute_type == EXCUTE_TYPE_SEND_RECEIVE) {
            para_info[task_it->send_rank][j].src_dest_info.src_valid = SEND_RECV_VALID;
            para_info[task_it->send_rank][j].src_dest_info.dest_rank = task_it->recv_rank;
            para_info[task_it->send_rank][j].src_dest_info.tag = HCCL_TAG_ANY;
            para_info[task_it->recv_rank][j].src_dest_info.dest_valid = SEND_RECV_VALID;
            para_info[task_it->recv_rank][j].src_dest_info.src_rank = task_it->send_rank;
            para_info[task_it->recv_rank][j].src_dest_info.tag = HCCL_TAG_ANY;
        }

        j++;
    }

    /* 每个rank用一个线程执行，每个rank操作并行执行 */
    if (run_type == TASK_RUN_PARALLEL) {
        for (s32 i = 0; i < device_count; i++) {
            para_info[i][0].task_num = task_vec.size();
            ;
            temp_tid = sal_thread_create("rank#i thread", excute_task, (void*)para_info[i]);
            EXPECT_NE(temp_tid, (sal_thread_t)NULL);

            if (temp_tid == (sal_thread_t)NULL) {
                return HCCL_E_THREAD;
            }

            rank_list[i].tids = temp_tid;
        }
    }

    /* 等待任务执行结束 */
    hccl_ret = wait_all_task_finish();
    TEST_CHECK_RET_VALUE(hccl_ret, HCCL_E_INTERNAL);

    free_two_dimension_memory(para_info, device_count);

    return HCCL_SUCCESS;
}

HcclResult tester::run(task_run_type_t runtype)
{
    HcclResult hccl_ret = HCCL_SUCCESS;

    run_type = runtype;

    /* 根据运行类型分发 */
    switch (runtype) {
        case TASK_RUN_PARALLEL:
            return run_by_thread();
            break;

        case TASK_RUN_SERIAL:
            return HCCL_E_NOT_SUPPORT;
            break;

        default:
            return HCCL_E_NOT_SUPPORT;
    }

    return hccl_ret;
}

HcclResult tester::dev_to_host_mem_synchronize()
{
    rtError_t rt_ret = RT_ERROR_NONE;
    s32 data_len;
    s64 sendbuf_len;
    s64 recvbuf_len;

    vector<test_task>::iterator task_it;

    for (task_it = task_vec.begin(); task_it != task_vec.end(); task_it++) {
        /* 将host_sendbuf信息异步copy到设备内存中 */
        /* 需要初始赋值的数据个数 */
        get_len_by_datatype(task_it->data_type, &data_len);

        if (task_it->excute_type == EXCUTE_TYPE_BROADCAST) {
            sendbuf_len = (s64)task_it->sendbuf_num * data_len;

            /* 将sendbuf从设备内存复制到host内存 */
            for (int rank = 0; rank < device_count; rank++) {
                if (rank == task_it->root) {
                    continue;
                }

                rt_ret = aclrtSetDevice(rank_list[rank].devid);
                EXPECT_EQ(rt_ret, RT_ERROR_NONE);

                if (rt_ret != RT_ERROR_NONE) {
                    STUB_ERROR("aclrtSetDevice failed, rank[%d], device id[%d]", rank, rank_list[rank].devid);
                    return HCCL_E_INTERNAL;
                }

                rt_ret = aclrtMemcpyAsync(
                    task_it->host_sendbuf[rank], (u64)sendbuf_len, task_it->dev_sendbuf[rank], (u64)sendbuf_len,
                    ACL_MEMCPY_DEVICE_TO_HOST, rank_list[rank].stream);

                if (rt_ret != ACL_SUCCESS) {
                    STUB_ERROR(
                        "dev_to_host_mem_synchronize rt_memcpy_async failed, rank[%d], dev_id[%d], rt_ret[%d]", rank,
                        rank_list[rank].devid, rt_ret);
                    return HCCL_E_INTERNAL;
                }
            }
        }

        recvbuf_len = (s64)task_it->recvbuf_num * data_len;

        /* 将sendbuf从设备内存复制到host内存 */
        for (int rank = 0; rank < device_count; rank++) {
            rt_ret = aclrtSetDevice(rank_list[rank].devid);
            EXPECT_EQ(rt_ret, RT_ERROR_NONE);

            if (rt_ret != RT_ERROR_NONE) {
                STUB_ERROR("aclrtSetDevice failed, rank[%d], device id[%d]", rank, rank_list[rank].devid);
                return HCCL_E_INTERNAL;
            }

            rt_ret = aclrtMemcpyAsync(
                task_it->host_recvbuf[rank], (u64)recvbuf_len, task_it->dev_recvbuf[rank], (u64)recvbuf_len,
                ACL_MEMCPY_DEVICE_TO_HOST, rank_list[rank].stream);

            if (rt_ret != ACL_SUCCESS) {
                STUB_ERROR(
                    "dev_to_host_mem_synchronize rt_memcpy_async failed, rank[%d], dev_id[%d], rt_ret[%d]", rank,
                    rank_list[rank].devid, rt_ret);
                return HCCL_E_INTERNAL;
            }
        }
    }

    /* 获取stream的操作的同步信号量, 等待流队列执行结束 */
    for (s32 rank = 0; rank < device_count; rank++) {
        rt_ret = aclrtSetDevice(rank_list[rank].devid);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        if (rt_ret != RT_ERROR_NONE) {
            STUB_ERROR("aclrtSetDevice failed, rank[%d], device id[%d]", rank, rank_list[rank].devid);
            return HCCL_E_INTERNAL;
        }

        rt_ret = aclrtSynchronizeStream(rank_list[rank].stream);
        EXPECT_EQ(rt_ret, ACL_SUCCESS);

        if (rt_ret != ACL_SUCCESS) {
            STUB_ERROR(
                "aclrtSynchronizeStream failed [%d], rank[%d], device id[%d]", hccl_ret, rank, rank_list[rank].devid);
            return hccl_ret;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult tester::wait_all_task_finish()
{
    rtError_t rt_ret = RT_ERROR_NONE;
    HcclResult hccl_ret = HCCL_SUCCESS;
    s32 sret = 0;

    for (s32 i = 0; i < device_count; ++i) {
        /* 等待线程运行结束 */
        while (sal_thread_is_running(rank_list[i].tids)) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    /* host的sendbuf和recvbuf同步dev上的sendbuf和recvbuf，用于后续的计算和比较 */
    hccl_ret = dev_to_host_mem_synchronize();

    if (hccl_ret != HCCL_SUCCESS) {
        STUB_ERROR("wait_all_task_finish dev_to_host_mem_synchronize failed ret[%d]", hccl_ret);
    }

    /* 获取stream的操作的同步信号量, 等待流队列执行结束 */
    for (s32 j = 0; j < device_count; j++) {
        rt_ret = aclrtSetDevice(rank_list[j].devid);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        if (rt_ret != RT_ERROR_NONE) {
            hccl_ret = HCCL_E_INTERNAL;
            STUB_ERROR(
                "wait_all_task_finish set device failed,rank[%d], dev_id[%d], rt_ret[%d]", j, rank_list[j].devid,
                rt_ret);
            goto cleanup;
        }

        rt_ret = aclrtSynchronizeStream(rank_list[j].stream);
        EXPECT_EQ(rt_ret, ACL_SUCCESS);

        if (rt_ret != ACL_SUCCESS) {
            STUB_ERROR(
                "wait_all_task_finish aclrtSynchronizeStream ,rank[%d], dev_id[%d], stream[%p], hccl_ret[%d]", j,
                rank_list[j].devid, rank_list[j].stream, hccl_ret);
            goto cleanup;
        }
    }

cleanup:

    /* 运行完毕之后，销毁线程 */
    for (s32 i = 0; i < device_count; ++i) {
        sret = sal_thread_destroy(rank_list[i].tids);
        EXPECT_EQ(sret, 0);

        if (sret != 0) {
            STUB_ERROR(
                "wait_all_task_finish sal_thread_destroy failed ,rank[%d], dev_id[%d], ret[%d]", i, rank_list[i].devid,
                sret);
            hccl_ret = HCCL_E_INTERNAL;
        }
    }

    return hccl_ret;
}

template <typename T>
HcclResult tester::compute_result(test_task_t* task, T type_para)
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    /* 根据操作类型，计算预期结果 */
    switch (task->excute_type) {
            /* broadcast的预期结果为ROOT节点的sendbuf值 */
        case EXCUTE_TYPE_BROADCAST:

            for (s32 offset = 0; offset < task->count; offset++) {
                ((T*)task->result_buf)[offset] = ((T*)task->host_sendbuf[task->root])[offset];
            }

            break;

            /* reduce和allreduce的预期结果为reduce的结果，区别是allreduce所有节点都是预期值 */
        case EXCUTE_TYPE_REDUCE:
        case EXCUTE_TYPE_ALL_REDUCE:
            (void)sal_memset(task->result_buf, task->count, 0, task->count);

            for (s32 offset = 0; offset < task->count; ++offset) {
                ((T*)task->result_buf)[offset] = ((T*)task->host_sendbuf[0])[offset];

                for (s32 rank = 1; rank < device_count; ++rank) {
                    switch (task->reduce_op) {
                        case HCCL_REDUCE_SUM:
                            ((T*)task->result_buf)[offset] += ((T*)task->host_sendbuf[rank])[offset];
                            break;

                        case HCCL_REDUCE_PROD:
                            ((T*)task->result_buf)[offset] *= ((T*)task->host_sendbuf[rank])[offset];
                            break;

                        case HCCL_REDUCE_MAX:
                            ((T*)task->result_buf)[offset]
                                = ((((T*)task->result_buf)[offset] > ((T*)task->host_sendbuf[rank])[offset]) ?
                                       ((T*)task->result_buf)[offset] :
                                       ((T*)task->host_sendbuf[rank])[offset]);
                            break;

                        case HCCL_REDUCE_MIN:
                            ((T*)task->result_buf)[offset]
                                = ((((T*)task->result_buf)[offset] < ((T*)task->host_sendbuf[rank])[offset]) ?
                                       ((T*)task->result_buf)[offset] :
                                       ((T*)task->host_sendbuf[rank])[offset]);
                    }
                }
            }

            break;

            /* reduce_scatter的预期值先使用reduce的结果，比较结果时会针对每个节点做scatter的操作 */
        case EXCUTE_TYPE_REDUCE_SCATTER:

            /*  注意: reduce_scatter操作的count是指scatter后的数据长度，运算前的数据长度为buf_size*device_count */
            (void)sal_memset(task->result_buf, task->count * device_count, 0, task->count * device_count);

            for (s32 offset = 0; offset < task->count * device_count; ++offset) {
                ((T*)task->result_buf)[offset] = ((T*)task->host_sendbuf[0])[offset];

                for (s32 rank = 1; rank < device_count; ++rank) {
                    switch (task->reduce_op) {
                        case HCCL_REDUCE_SUM:
                            ((T*)task->result_buf)[offset] += ((T*)task->host_sendbuf[rank])[offset];
                            break;

                        case HCCL_REDUCE_PROD:
                            ((T*)task->result_buf)[offset] *= ((T*)task->host_sendbuf[rank])[offset];
                            break;

                        case HCCL_REDUCE_MAX:
                            ((T*)task->result_buf)[offset]
                                = ((((T*)task->result_buf)[offset] > ((T*)task->host_sendbuf[rank])[offset]) ?
                                       ((T*)task->result_buf)[offset] :
                                       ((T*)task->host_sendbuf[rank])[offset]);
                            break;

                        case HCCL_REDUCE_MIN:
                            ((T*)task->result_buf)[offset]
                                = ((((T*)task->result_buf)[offset] < ((T*)task->host_sendbuf[rank])[offset]) ?
                                       ((T*)task->result_buf)[offset] :
                                       ((T*)task->host_sendbuf[rank])[offset]);
                    }
                }
            }

            break;

            /* all_gather的预期结果使用gather后的结果 */
        case EXCUTE_TYPE_ALL_GATHER:

            for (s32 rank = 0; rank < device_count; rank++) {
                for (s32 offset = 0; offset < task->count; offset++) {
                    ((T*)task->result_buf)[rank * task->count + offset] = ((T*)task->host_sendbuf[rank])[offset];
                }
            }

            break;

            /* send_receive的预期值是sendbuf的数据 */
        case EXCUTE_TYPE_SEND_RECEIVE:

            // send_receive的root是复用的，计算result_buf表示发送端，比较结果时表示接收端
            for (s32 offset = 0; offset < task->count; offset++) {
                ((T*)task->result_buf)[offset] = ((T*)task->host_sendbuf[task->send_rank])[offset];
            }

            break;
    }

    return HCCL_SUCCESS;
}

template <typename T>
HcclResult tester::print_buf_info(test_task_t* task, T cpu_type)
{
    rtError_t rt_ret = RT_ERROR_NONE;
    s32 recv_data_num = task->count;
    s32 rslt_data_num;
    s32 send_data_num;
    s32 send_rank = INVALID_RANK;
    s32 recv_rank = INVALID_RANK;
    char format_float[] = "%8g ";
    char format_int[] = "%02X ";
    char* format = ((typeid(T).name()) == (typeid(1.0f).name()) ? format_float : format_int);
    char disp_str[LOG_TMPBUF_SIZE] = {0};
    s32 offset = 0;

    char op_name[EXCUTE_TYPE_RESERVED][20] = {
        "BROADCAST", "REDUCE", "ALL_GATHER", "REDUCE_SCATTER", "ALL_REDUCE", "SEND_RECEIVE",
    };

    if (task->buf_print_enable) {
        printf("\r\n---->>---->> operation type: %s <<----<<----\n", op_name[task->excute_type]);
    }

    /* 计算sendbuf和resultbuf的数据个数 */
    if (task->excute_type == EXCUTE_TYPE_REDUCE_SCATTER) {
        send_data_num = task->count * device_count;
    } else {
        send_data_num = task->count;
    }

    if (task->excute_type == EXCUTE_TYPE_ALL_GATHER || task->excute_type == EXCUTE_TYPE_REDUCE_SCATTER) {
        rslt_data_num = task->count * device_count;
    } else {
        rslt_data_num = task->count;
    }

    if (task->excute_type == EXCUTE_TYPE_REDUCE) {
        recv_rank = task->root;
    }

    if (task->excute_type == EXCUTE_TYPE_SEND_RECEIVE) {
        recv_rank = task->recv_rank;
    }

    if (task->excute_type == EXCUTE_TYPE_BROADCAST || task->excute_type == EXCUTE_TYPE_SEND_RECEIVE) {
        send_rank = task->root;
    }

    /* 打印sendbuf数据 */
    if (task->buf_print_enable & PRINT_MASK_SEND) {
        printf("---->>send buf data num: %d\n", send_data_num);

        for (s32 rank = 0; rank < device_count; rank++) {
            /* 如果有root节点，只打印root节点的recv buf */
            if ((send_rank != INVALID_RANK) && (rank != send_rank)) {
                continue;
            }

            printf("rank[%d]:\t", rank);

            for (s32 offset = 0; offset < send_data_num; offset++) {
                printf(format, ((T*)task->dev_sendbuf[rank])[offset]);
            }

            printf("\n");
        }
    }

    /* 打印recv_buf数据 */
    if (task->buf_print_enable & PRINT_MASK_RECV && task->excute_type != EXCUTE_TYPE_BROADCAST) {
        printf("---->>recv buf data num: %d\n", recv_data_num);

        for (s32 rank = 0; rank < device_count; rank++) {
            /* 如果有root节点，只打印root节点的recv buf */
            if ((recv_rank != INVALID_RANK) && (rank != recv_rank)) {
                continue;
            }

            printf("rank[%d]:\t", rank);

            for (s32 offset = 0; offset < recv_data_num; offset++) {
                printf(format, ((T*)task->host_recvbuf[rank])[offset]);
            }

            printf("\n");
        }
    }

    /* 打印result_buf数据 */
    if (task->buf_print_enable & PRINT_MASK_RSLT) {
        printf("---->>result buf data num: %d\n", rslt_data_num);

        for (s32 offset = 0; offset < rslt_data_num; offset++) {
            printf(format, ((T*)task->result_buf)[offset]);
        }

        printf("\r\n");
    }

    return HCCL_SUCCESS;
}

template <typename T>
HcclResult tester::compare_result(test_task_t* task, T cpu_type)
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    s32 err_count = 0;
    s32 result_rankid;
    s8 temp = 0;

    switch (task->excute_type) {
            /* broadcast和all_reduce操作比较每一个sendbuf是否都和预期值一致 */
        case EXCUTE_TYPE_BROADCAST:
            for (s32 rank = 0; rank < device_count; rank++) {
                for (s32 offset = 0; offset < task->count; offset++) {
                    /* 模板类型为float型 */
                    if ((typeid(T).name()) == (typeid(1.0f).name())) {
                        if (fabs((((T*)task->host_sendbuf[rank])[offset]) - ((T*)task->result_buf)[offset])
                            > FLOAT_MAX_DIFF_RANGE) {
                            err_count++;
                        }
                    } else {
                        if (((T*)task->result_buf)[offset] != ((T*)task->host_sendbuf[rank])[offset]) {
                            err_count++;
                        }
                    }
                }
            }

            break;

            /* broadcast和all_reduce操作比较每一个sendbuf是否都和预期值一致 */
        case EXCUTE_TYPE_ALL_REDUCE:
        case EXCUTE_TYPE_ALL_GATHER:
            for (s32 rank = 0; rank < device_count; rank++) {
                for (s32 offset = 0; offset < task->count; offset++) {
                    /* 模板类型为float型 */
                    if ((typeid(T).name()) == (typeid(1.0f).name())) {
                        if (fabs((((T*)task->host_recvbuf[rank])[offset]) - ((T*)task->result_buf)[offset])
                            > FLOAT_MAX_DIFF_RANGE) {
                            err_count++;
                        }
                    } else {
                        if (((T*)task->result_buf)[offset] != ((T*)task->host_recvbuf[rank])[offset]) {
                            err_count++;
                        }
                    }
                }
            }

            break;

            /* reduce和send_receive操作会比较root节点或接收节点是否和预期一致 */
        case EXCUTE_TYPE_REDUCE:
        case EXCUTE_TYPE_SEND_RECEIVE:

            result_rankid = ((task->excute_type == EXCUTE_TYPE_REDUCE) ? task->root : task->recv_rank);

            for (s32 offset = 0; offset < task->count; ++offset) {
                /* 模板类型为float型 */
                if ((typeid(T).name()) == (typeid(1.0f).name())) {
                    if (fabs((((T*)task->host_recvbuf[result_rankid])[offset]) - ((T*)task->result_buf)[offset])
                        > FLOAT_MAX_DIFF_RANGE) {
                        err_count++;
                    }
                } else {
                    if (((T*)task->result_buf)[offset] != ((T*)task->host_recvbuf[result_rankid])[offset]) {
                        err_count++;
                    }
                }
            }

            break;

        case EXCUTE_TYPE_REDUCE_SCATTER:

            for (s32 rank = 0; rank < device_count; ++rank) {
                for (s32 offset = 0; offset < task->count; offset++) {
                    /* 模板类型为float型 */
                    if ((typeid(T).name()) == (typeid(1.0f).name())) {
                        if (fabs(
                                ((T*)task->host_recvbuf[rank])[offset]
                                - ((T*)task->result_buf)[rank * task->count + offset])
                            > FLOAT_MAX_DIFF_RANGE) {
                            err_count++;
                        }
                    } else {
                        if (((T*)task->result_buf)[rank * task->count + offset]
                            != ((T*)task->host_recvbuf[rank])[offset]) {
                            err_count++;
                        }
                    }
                }
            }

            break;
    }

    EXPECT_EQ(err_count, 0);

    if (err_count == 0) {
        return HCCL_SUCCESS;
    } else {
        return HCCL_E_INTERNAL;
    }
}

template <typename T>
HcclResult tester::check_single_task(test_task_t* task, T cpu_type)
{
    HcclResult hccl_ret = HCCL_SUCCESS;

    /* 计算预期结果 */
    hccl_ret = compute_result(task, cpu_type);
    EXPECT_EQ(HCCL_SUCCESS, hccl_ret);
    if (HCCL_SUCCESS != hccl_ret) {
        STUB_ERROR("compute_result failed[%d]", hccl_ret);
    }

    /* 打印buff信息 */
    hccl_ret = print_buf_info(task, cpu_type);
    EXPECT_EQ(HCCL_SUCCESS, hccl_ret);
    if (HCCL_SUCCESS != hccl_ret) {
        STUB_ERROR("print_buf_info failed[%d]", hccl_ret);
    }

    /* 比较结果 */
    hccl_ret = compare_result(task, cpu_type);
    EXPECT_EQ(HCCL_SUCCESS, hccl_ret);
    if (HCCL_SUCCESS != hccl_ret) {
        STUB_ERROR("compare_result failed[%d]", hccl_ret);
    }

    /* destroy */
    hccl_ret = excute_free_task_mem(*task);
    EXPECT_EQ(HCCL_SUCCESS, hccl_ret);
    if (HCCL_SUCCESS != hccl_ret) {
        STUB_ERROR("excute_free_task_mem failed[%d]", hccl_ret);
    }

    return hccl_ret;
}

HcclResult tester::check_result()
{
    HcclResult hccl_ret = HCCL_SUCCESS;
    float cpu_tpye_float = 0.0f;
    s8 cpu_tpye_s8 = 0;

    /* 取出测试任务队列中的excute，顺序执行 */
    vector<test_task>::iterator task_it;

    for (task_it = task_vec.begin(); task_it != task_vec.end(); task_it++) {
        if (task_it->data_type == HCCL_DATA_TYPE_FP32) {
            hccl_ret = check_single_task(&(*task_it), cpu_tpye_float);
        } else {
            hccl_ret = check_single_task(&(*task_it), cpu_tpye_s8);
        }

        if (hccl_ret != HCCL_SUCCESS) {
            STUB_ERROR("task failed");
        }
    }

    hccl_ret = destroy_comms();

    return HCCL_SUCCESS;
}

HcclResult tester::destroy_comms()
{
    rtError_t rt_ret = RT_ERROR_NONE;

    // 释放资源
    for (s32 rank = 0; rank < device_count; ++rank) {
        rt_ret = aclrtSetDevice(rank_list[rank].devid);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(rank_list[rank].stream);
        EXPECT_EQ(rt_ret, ACL_SUCCESS);

        HcclCommDestroy(rank_list[rank].comm);
    }

    return HCCL_SUCCESS;
}
