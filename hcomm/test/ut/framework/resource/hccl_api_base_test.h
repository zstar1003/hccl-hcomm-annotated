/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <securec.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/types.h>
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "externalinput.h"
#include <nlohmann/json.hpp>
#include "rt_external.h"
#include "launch_device.h"

#define private public
#define protected public
#include "topoinfo_detect.h"
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "coll_batch_send_recv_executor.h"
#include "coll_reduce_scatter_v_executor.h"
#include "coll_all_gather_v_executor.h"
#include "preempt_port_manager.h"
#include "task_abort_handler_pub.h"
#undef protected
#undef private
#include "hccl_whitelist.h"
#include "profiling_manager.h"
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include "llt_hccl_stub_pub.h"
#include <iostream>
#include <fstream>
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "dltdt_function.h"
#include "topoinfo_ranktableParser_pub.h"
#include "v80_rank_table.h"
#include "externalinput_pub.h"
#include "op_base.h"
#include "param_check_pub.h"
#include "comm_config_pub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exception_handler.h"
#include "plugin_runner_pub.h"
#include "task_exception_handler_pub.h"
#include "topoinfo_exchange_dispatcher.h"
#include "heartbeat.h"
#include <dirent.h>
#include "hccl/hccl_res.h"

using namespace std;

using namespace hccl;

inline void UtInitHcclCommConfig(HcclCommConfig& config) { HcclCommConfigInit(&config); }

// 1KB
#define HCCL_COM_DATA_SIZE 1024

// 300MB
#define HCCL_COM_BIG_DATA_SIZE (300 * 1024 * 1024)

// 本测试中绝大多数测试用例的父类，目的是减少冗余代码
class BaseInit : public testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    char rankTableFileName[24];
    HcclComm comm;
    rtStream_t stream;
    int rankNum;
};

// 设置device，判断返回值是否为成功
void Ut_Device_Set(int devId);

// 根据rankTable和filename，创建对应的配置文件，用于通信域初始化
void Ut_Clusterinfo_File_Create(const char* filename, nlohmann::json rankTable);

// 建立Root节点的拓扑探测，获取rootinfo（如果想修改root的ip，需要设置环境变量），检测返回值是否为成功。
HcclRootInfo Ut_Get_Root_Info(int devId);

// 销毁通信域，检测返回值是否为成功
void Ut_Comm_Destroy(void*& comm);

// 设置device和创建通信域，检测返回值是否为成功
void Ut_Comm_Create(void*& comm, int devId, const char* rankTableFile, int rankId);

// 创建内存,并将内存赋值为零
void Ut_Buf_Create(s8*& buf, int len);

// 用于不均分发送buf的API接口的测试，仅用于简单测试，只对buf进行均分
void Ut_BufV_Create(s8*& buf, int bufLen, u64*& counts, int countsLen, int c, u64*& displs, int displsLen, int d);

// 创建stream，检测返回值是否为成功
void Ut_Stream_Create(rtStream_t& stream, int priority);

// stream同步，检测返回值是否为成功
void Ut_Stream_Synchronize(rtStream_t& stream);

// 销毁stream，检测返回值是否为成功
void Ut_Stream_Destroy(rtStream_t& stream);

// stream 同步之后销毁，检测返回值是否为成功
void Ut_Stream_SynchronizeAndDestroy(rtStream_t& stream);

// 根据ranktable，创建多机通信的配置文件，并mock掉必须的函数。
void Ut_MultiServer_MOCK_And_Clusterinfo_File_Create(const char* filename, nlohmann::json rankTable);

// mock掉必须函数，保证HcclGetRootInfo可以被正常测试
void When_Need_HcclGetRootInfo(void);

// 创建名为rankTableFileName的ranktable文件，具体内容为1个主机，携带了1张卡
#define UT_USE_RANK_TABLE_910_1SERVER_1RANK                                          \
    do {                                                                             \
        Ut_Clusterinfo_File_Create(rankTableFileName, rank_table_910_1server_1rank); \
    } while (0)

// 创建名为rankTableFileName的ranktable文件，具体内容为1个主机，携带了2张卡
#define UT_USE_RANK_TABLE_910_1SERVER_2RANK                                                                 \
    do {                                                                                                    \
        MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));                       \
        MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS)); \
        Ut_Clusterinfo_File_Create(rankTableFileName, rank_table_910_1server_2rank);                        \
    } while (0)

// 创建名为rankTableFileName的ranktable文件，具体内容为2个主机，分别携带4张卡
#define UT_USE_RANK_TABLE_910_2SERVER_4RANK                                                               \
    do {                                                                                                  \
        Ut_MultiServer_MOCK_And_Clusterinfo_File_Create(rankTableFileName, rank_table_910_2server_4rank); \
    } while (0)
/*
    根据测试用例名称进行判断，如果用例名中出现1Server2Rank，那就选择UT_USE_RANK_TABLE_910_1SERVER_2RANK，否则
    如果出现了2Server4Rank那就选择UT_USE_RANK_TABLE_910_2SERVER_4RANK，如果都没有，那就默认为
    UT_USE_RANK_TABLE_910_1SERVER_1RANK
*/
#define UT_USE_1SERVER_1RANK_AS_DEFAULT                                       \
    do {                                                                      \
        auto* info = ::testing::UnitTest::GetInstance()->current_test_info(); \
        if (info) {                                                           \
            std::string name = info->name();                                  \
            if (name.find("1Server2Rank") != std::string::npos) {             \
                UT_USE_RANK_TABLE_910_1SERVER_2RANK;                          \
                rankNum = 2;                                                  \
            } else if (name.find("2Server4Rank") != std::string::npos) {      \
                UT_USE_RANK_TABLE_910_2SERVER_4RANK;                          \
                rankNum = 8;                                                  \
            } else {                                                          \
                UT_USE_RANK_TABLE_910_1SERVER_1RANK;                          \
                rankNum = 1;                                                  \
            }                                                                 \
        } else {                                                              \
            UT_USE_RANK_TABLE_910_1SERVER_1RANK;                              \
            rankNum = 1;                                                      \
        }                                                                     \
    } while (0)

/*
    根据测试用例名称进行判断，如果用例名中出现了2Server4Rank那就选择UT_USE_RANK_TABLE_910_2SERVER_4RANK，
    如果没有，那就默认为UT_USE_RANK_TABLE_910_1SERVER_1RANK
*/
#define UT_USE_1SERVER_2RANK_AS_DEFAULT                                       \
    do {                                                                      \
        auto* info = ::testing::UnitTest::GetInstance()->current_test_info(); \
        if (info) {                                                           \
            std::string name = info->name();                                  \
            if (name.find("2Server4Rank") != std::string::npos) {             \
                UT_USE_RANK_TABLE_910_2SERVER_4RANK;                          \
                rankNum = 8;                                                  \
            } else {                                                          \
                UT_USE_RANK_TABLE_910_1SERVER_2RANK;                          \
                rankNum = 2;                                                  \
            }                                                                 \
        } else {                                                              \
            UT_USE_RANK_TABLE_910_1SERVER_2RANK;                              \
            rankNum = 2;                                                      \
        }                                                                     \
    } while (0)
/*
    对sendBuf、recvBuf和count进行初始化，该宏主要用于函数HcclAllGather、HcclAllReduce、HcclReduce、
    HcclReduceScatter、HcclScatter的初始化变量
*/
#define UT_SET_SENDBUF_RECVBUF_COUNT(sendSize, recvSize, countSize) \
    do {                                                            \
        Ut_Buf_Create(sendBuf, sendSize);                           \
        Ut_Buf_Create(recvBuf, recvSize);                           \
        count = countSize;                                          \
    } while (0)

// 对sendBuf、sendCount、recvBuf和recvCount进行初始化，该宏主要用于函数HcclAlltoAll的初始化变量
#define UT_SET_SENDBUF_COUNT_RECVBUF_COUNT(sendSize, sendCountSize, recvSize, recvCountSize) \
    do {                                                                                     \
        Ut_Buf_Create(sendBuf, sendSize);                                                    \
        sendCount = sendCountSize;                                                           \
        Ut_Buf_Create(recvBuf, recvSize);                                                    \
        recvCount = recvCountSize;                                                           \
    } while (0)

// 用于对sendBuf、recvBuf的销毁，可与UT_SET_SENDBUF_RECVBUF_COUNT或UT_SET_SENDBUF_COUNT_RECVBUF_COUNT配套使用
#define UT_UNSET_SENDBUF_RECVBUF() \
    do {                           \
        sal_free(sendBuf);         \
        sal_free(recvBuf);         \
    } while (0)

// 对sendBuf、count进行初始化，该宏主要用于函数HcclBoardcast、HcclSend的初始化变量
#define UT_SET_SENDBUF_COUNT(sendSize, countSize) \
    do {                                          \
        Ut_Buf_Create(sendBuf, sendSize);         \
        count = countSize;                        \
    } while (0)

// 用于对sendBuf的销毁，可与UT_SET_SENDBUF_COUNT配套使用
#define UT_UNSET_SENDBUF() \
    do {                   \
        sal_free(sendBuf); \
    } while (0)

// 对recvBuf、count进行初始化，该宏主要用于函数HcclRecv的初始化变量
#define UT_SET_RECVBUF_COUNT(recvSize, countSize) \
    do {                                          \
        Ut_Buf_Create(recvBuf, recvSize);         \
        count = countSize;                        \
    } while (0)

// 用于对recvBuf的销毁，可与UT_SET_RECVBUF_COUNT配套使用
#define UT_UNSET_RECVBUF() \
    do {                   \
        sal_free(recvBuf); \
    } while (0)

// 对sendBuf、sendCounts、sendDispls、recvBuf、recvCount进行初始化，该宏主要用于函数HcclReduceScatterV的初始化变量
#define UT_SET_SENDBUFV_RECVBUF_COUNT(                                                                   \
    sendSize, sendCountsSize, sendCountsValue, sendDisplsSize, sendDisplsValue, recvSize, recvCountSize) \
    do {                                                                                                 \
        Ut_BufV_Create(                                                                                  \
            sendBuf, sendSize, sendCounts, sendCountsSize, sendCountsValue, sendDispls, sendDisplsSize,  \
            sendDisplsValue);                                                                            \
        Ut_Buf_Create(recvBuf, recvSize);                                                                \
        recvCount = recvCountSize;                                                                       \
    } while (0)

// 对sendBuf、sendCounts、sendDispls、recvBuf的销毁，可与UT_SET_SENDBUFV_RECVBUF_COUNT配套使用
#define UT_UNSET_SENDBUFV_RECVBUF() \
    do {                            \
        sal_free(sendBuf);          \
        sal_free(sendCounts);       \
        sal_free(sendDispls);       \
        sal_free(recvBuf);          \
    } while (0)

// 对sendBuf、sendCount、recvBuf、recvCounts、recvDispls进行初始化，该宏主要用于函数HcclAllGatherV的初始化变量
#define UT_SET_SENDBUF_COUNT_RECVBUFV(                                                                   \
    sendSize, sendCountSize, recvSize, recvCountsSize, recvCountsValue, recvDisplsSize, recvDisplsValue) \
    do {                                                                                                 \
        Ut_Buf_Create(sendBuf, sendSize);                                                                \
        sendCount = sendCountSize;                                                                       \
        Ut_BufV_Create(                                                                                  \
            recvBuf, recvSize, recvCounts, recvCountsSize, recvCountsValue, recvDispls, recvDisplsSize,  \
            recvDisplsValue);                                                                            \
    } while (0)

// 对sendBuf、recvBuf、recvCounts、recvDispls的销毁，可与UT_SET_SENDBUF_COUNT_RECVBUFV配套使用
#define UT_UNSET_SENDBUF_RECVBUFV() \
    do {                            \
        sal_free(sendBuf);          \
        sal_free(recvBuf);          \
        sal_free(recvCounts);       \
        sal_free(recvDispls);       \
    } while (0)
/*
    对sendBuf、sendCounts、sendDispls、recvBuf、recvCounts、recvDispls进行初始化，该宏主要用于函数
    HcclAlltoAllV的初始化变量
*/
#define UT_SET_SENDBUFV_RECVBUFV(                                                                         \
    sendSize, sendCountsSize, sendCountsValue, sendDisplsSize, sendDisplsValue, recvSize, recvCountsSize, \
    recvCountsValue, recvDisplsSize, recvDisplsValue)                                                     \
    do {                                                                                                  \
        Ut_BufV_Create(                                                                                   \
            sendBuf, sendSize, sendCounts, sendCountsSize, sendCountsValue, sendDispls, sendDisplsSize,   \
            sendDisplsValue);                                                                             \
        Ut_BufV_Create(                                                                                   \
            recvBuf, recvSize, recvCounts, recvCountsSize, recvCountsValue, recvDispls, recvDisplsSize,   \
            recvDisplsValue);                                                                             \
    } while (0)

// 对sendBuf、sendCounts、sendDispls、recvBuf、recvCounts、recvDispls的销毁，可与UT_SET_SENDBUFV_RECVBUFV配套使用
#define UT_UNSET_SENDBUFV_RECVBUFV() \
    do {                             \
        sal_free(sendBuf);           \
        sal_free(sendCounts);        \
        sal_free(sendDispls);        \
        sal_free(recvBuf);           \
        sal_free(recvCounts);        \
        sal_free(recvDispls);        \
    } while (0)

// 以devId=0，rankId=0来创建通信域
#define UT_COMM_CREATE_DEFAULT(comm)                   \
    do {                                               \
        Ut_Comm_Create(comm, 0, rankTableFileName, 0); \
    } while (0)

// 以priority=0来创建stream
#define UT_STREAM_CREATE_DEFAULT(stream) \
    do {                                 \
        Ut_Stream_Create(stream, 0);     \
    } while (0)

// 调用UT_UNSET_SENDBUF_RECVBUF并销毁stream和comm，销毁前先对stream做一下同步
#define UT_UNSET_SENDBUF_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, Stream) \
    do {                                                                              \
        UT_UNSET_SENDBUF_RECVBUF();                                                   \
        Ut_Stream_SynchronizeAndDestroy(stream);                                      \
        Ut_Comm_Destroy(comm);                                                        \
    } while (0)

// 调用UT_UNSET_SENDBUF_RECVBUF并销毁stream和comm
#define UT_UNSET_SENDBUF_RECVBUF_COMM_STREAM(comm, Stream) \
    do {                                                   \
        UT_UNSET_SENDBUF_RECVBUF();                        \
        Ut_Stream_Destroy(stream);                         \
        Ut_Comm_Destroy(comm);                             \
    } while (0)

// 调用UT_UNSET_SENDBUF并销毁stream和comm，销毁前先对stream做一下同步
#define UT_UNSET_SENDBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, Stream) \
    do {                                                                      \
        UT_UNSET_SENDBUF();                                                   \
        Ut_Stream_SynchronizeAndDestroy(stream);                              \
        Ut_Comm_Destroy(comm);                                                \
    } while (0)

// 调用UT_UNSET_SENDBUF并销毁stream和comm
#define UT_UNSET_SENDBUF_COMM_STREAM(comm, Stream) \
    do {                                           \
        UT_UNSET_SENDBUF();                        \
        Ut_Stream_Destroy(stream);                 \
        Ut_Comm_Destroy(comm);                     \
    } while (0)

// 调用UT_UNSET_RECVBUF并销毁stream和comm，销毁前先对stream做一下同步
#define UT_UNSET_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, Stream) \
    do {                                                                      \
        UT_UNSET_RECVBUF();                                                   \
        Ut_Stream_SynchronizeAndDestroy(stream);                              \
        Ut_Comm_Destroy(comm);                                                \
    } while (0)

// 调用UT_UNSET_RECVBUF并销毁stream和comm
#define UT_UNSET_RECVBUF_COMM_STREAM(comm, Stream) \
    do {                                           \
        UT_UNSET_RECVBUF();                        \
        Ut_Stream_Destroy(stream);                 \
        Ut_Comm_Destroy(comm);                     \
    } while (0)

// 调用UT_UNSET_SENDBUFV_RECVBUF并销毁stream和comm，销毁前先对stream做一下同步
#define UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, Stream) \
    do {                                                                               \
        UT_UNSET_SENDBUFV_RECVBUF();                                                   \
        Ut_Stream_SynchronizeAndDestroy(stream);                                       \
        Ut_Comm_Destroy(comm);                                                         \
    } while (0)

// 调用UT_UNSET_SENDBUFV_RECVBUF并销毁stream和comm
#define UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM(comm, Stream) \
    do {                                                    \
        UT_UNSET_SENDBUFV_RECVBUF();                        \
        Ut_Stream_Destroy(stream);                          \
        Ut_Comm_Destroy(comm);                              \
    } while (0)

// 调用UT_UNSET_SENDBUF_RECVBUFV并销毁stream和comm，销毁前先对stream做一下同步
#define UT_UNSET_SENDBUF_RECVBUFV_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, Stream) \
    do {                                                                               \
        UT_UNSET_SENDBUF_RECVBUFV();                                                   \
        Ut_Stream_SynchronizeAndDestroy(stream);                                       \
        Ut_Comm_Destroy(comm);                                                         \
    } while (0)

// 调用UT_UNSET_SENDBUF_RECVBUFV并销毁stream和comm
#define UT_UNSET_SENDBUF_RECVBUFV_COMM_STREAM(comm, Stream) \
    do {                                                    \
        UT_UNSET_SENDBUF_RECVBUFV();                        \
        Ut_Stream_Destroy(stream);                          \
        Ut_Comm_Destroy(comm);                              \
    } while (0)

// 调用UT_UNSET_SENDBUFV_RECVBUFV并销毁stream和comm，销毁前先对stream做一下同步
#define UT_UNSET_SENDBUFV_RECVBUFV_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, Stream) \
    do {                                                                                \
        UT_UNSET_SENDBUFV_RECVBUFV();                                                   \
        Ut_Stream_SynchronizeAndDestroy(stream);                                        \
        Ut_Comm_Destroy(comm);                                                          \
    } while (0)

// 调用UT_UNSET_SENDBUFV_RECVBUFV并销毁stream和comm
#define UT_UNSET_SENDBUFV_RECVBUFV_COMM_STREAM(comm, Stream) \
    do {                                                     \
        UT_UNSET_SENDBUFV_RECVBUFV();                        \
        Ut_Stream_Destroy(stream);                           \
        Ut_Comm_Destroy(comm);                               \
    } while (0)

// 销毁stream和comm，销毁前先对stream做一下同步
#define UT_UNSET_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, Stream) \
    do {                                                              \
        Ut_Stream_SynchronizeAndDestroy(stream);                      \
        Ut_Comm_Destroy(comm);                                        \
    } while (0)

// 销毁stream和comm
#define UT_UNSET_COMM_STREAM(comm, Stream) \
    do {                                   \
        Ut_Stream_Destroy(stream);         \
        Ut_Comm_Destroy(comm);             \
    } while (0)
