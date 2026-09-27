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
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>

#define private public
#define protected public
#include "hccl_impl.h"
#include "hccl_comm_pub.h"
#undef protected
#undef private

#include "llt_hccl_stub_pub.h"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "hccl/base.h"
#include "hccl/hccl_ex.h"
#include <hccl/hccl_types.h>
#include "topoinfo_ranktableParser_pub.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include <unistd.h>
#include "externalinput_pub.h"
#include "v80_rank_table.h"
#include "externalinput.h"
#include "op_base.h"
#include <functional>
#include <map>

using namespace std;
using namespace hccl;

class OpbaseMultiThreadTest : public testing::TestWithParam<bool> {
protected:
    virtual void SetUp()
    {
        ra_set_test_type(0, "ST_TEST");
        static s32 call_cnt = 0;
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        ResetInitState();
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        TsdClose(1);
        std::cout << "A Test TearDown" << std::endl;
    }
};

struct ThreadContext {
    HcclComm comm;
    int32_t deviceLogicID;
    uint32_t ndev;
};

#define HCCL_COM_DATA_SIZE 1024

void ExecAllReduce(int ndev, HcclComm comm, uint32_t deviceLogicID, uint32_t rankId)
{
    int ret = HCCL_SUCCESS;
    /* 1. 申请相关资源 */
    ret = hrtSetDevice(deviceLogicID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtStream_t stream;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    s32 count = HCCL_COM_DATA_SIZE;
    s8* sendbuf;
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

    s8* recvbuf;
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    uint32_t rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, ndev);

    uint32_t rankID = 0;
    ret = HcclGetRankId(comm, &rankID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankID, rankId);

    /* 2. 执行allreduce */
    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    /* 3. 校验执行结果准确性 */
    s32 errors = 0;
    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }
    EXPECT_EQ(errors, 0);

    /* 4. 释放相关资源 */
    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ret = hrtResetDevice(deviceLogicID);
    EXPECT_EQ(ret, 0);
    return;
}
#if 0 // 执行失败Unknown comm devType
TEST_F(OpbaseMultiThreadTest, ut_HcclAllReduce)
{
    int ret = HCCL_SUCCESS;
    uint32_t ndev = 1;
    int32_t devices[ndev] = {0};
    HcclComm comms[ndev];
    for (int i = 0; i < ndev; i++) {
        ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, 0);

    std::vector<std::thread> threads;
    threads.resize(ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i] = std::thread(ExecAllReduce, ndev, comms[i], devices[i], i);
    }

    for (uint32_t i = 0; i < ndev; ++i) {
        threads[i].join();
    }

    for (uint32_t i = 0; i < ndev; i++) {
        ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
        ret = HcclCommDestroy(comms[i]);
        EXPECT_EQ(ret, 0);
    }
}
#endif

void ExecAllGather(int ndev, HcclComm comm, uint32_t deviceLogicID, uint32_t rankId)
{
    int ret = HCCL_SUCCESS;
    /* 1. 申请相关资源 */
    ret = hrtSetDevice(deviceLogicID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtStream_t stream;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    s32 count = 8;
    s8* sendbuf;
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

    s8* recvbuf;
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    uint32_t rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, ndev);

    uint32_t rankID = 0;
    ret = HcclGetRankId(comm, &rankID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankID, rankId);

    /* 2. 执行allreduce */
    ret = HcclAllGatherInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    /* 3. 校验执行结果准确性 */
    s32 errors = 0;
    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }
    EXPECT_EQ(errors, 0);

    /* 4. 释放相关资源 */
    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ret = hrtResetDevice(deviceLogicID);
    EXPECT_EQ(ret, 0);
    return;
}
#if 0 // 执行失败Unknown comm devType
TEST_F(OpbaseMultiThreadTest, ut_HcclAllGather)
{
    int ret = HCCL_SUCCESS;
    uint32_t ndev = 1;
    int32_t devices[ndev] = {0};
    HcclComm comms[ndev];
    for (int i = 0; i < ndev; i++) {
        ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, 0);

    std::vector<std::thread> threads;
    threads.resize(ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i] = std::thread(ExecAllGather, ndev, comms[i], devices[i], i);
    }

    for (uint32_t i = 0; i < ndev; ++i) {
        threads[i].join();
    }

    for (uint32_t i = 0; i < ndev; i++) {
        ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
        ret = HcclCommDestroy(comms[i]);
        EXPECT_EQ(ret, 0);
    }
}
#endif
void ExecBroadCast(int ndev, HcclComm comm, uint32_t deviceLogicID, uint32_t rankId)
{
    int ret = HCCL_SUCCESS;
    /* 1. 申请相关资源 */
    ret = hrtSetDevice(deviceLogicID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtStream_t stream;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    s32 count = HCCL_COM_DATA_SIZE;

    s8* sendbuf;
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    uint32_t rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, ndev);

    uint32_t rankID = 0;
    ret = HcclGetRankId(comm, &rankID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankID, rankId);

    /* 2. 执行 HcclBroadcastInner */
    ret = HcclBroadcastInner(sendbuf, count, HCCL_DATA_TYPE_INT8, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    /* 3. 校验执行结果准确性 */
    s32 errors = 0;
    for (int j = 0; j < count; j++) {
        if (sendbuf[j] != 2) {
            errors++;
            break;
        }
    }
    EXPECT_EQ(errors, 0);

    /* 4. 释放相关资源 */
    sal_free(sendbuf);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ret = hrtResetDevice(deviceLogicID);
    EXPECT_EQ(ret, 0);
    return;
}
#if 0 // 执行失败Unknown comm devType
TEST_F(OpbaseMultiThreadTest, ut_HcclBroadCast)
{
    int ret = HCCL_SUCCESS;
    uint32_t ndev = 1;
    int32_t devices[ndev] = {0};
    HcclComm comms[ndev];
    for (int i = 0; i < ndev; i++) {
        ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, 0);

    std::vector<std::thread> threads;
    threads.resize(ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i] = std::thread(ExecBroadCast, ndev, comms[i], devices[i], i);
    }

    for (uint32_t i = 0; i < ndev; ++i) {
        threads[i].join();
    }

    for (uint32_t i = 0; i < ndev; i++) {
        ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
        ret = HcclCommDestroy(comms[i]);
        EXPECT_EQ(ret, 0);
    }
}
#endif

void ExecReduceScatter(int ndev, HcclComm comm, uint32_t deviceLogicID, uint32_t rankId)
{
    int ret = HCCL_SUCCESS;
    /* 1. 申请相关资源 */
    ret = hrtSetDevice(deviceLogicID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtStream_t stream;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    s32 count = HCCL_COM_DATA_SIZE;

    s8* sendbuf;
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

    s8* recvbuf;
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    uint32_t rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, ndev);

    uint32_t rankID = 0;
    ret = HcclGetRankId(comm, &rankID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankID, rankId);

    /* 2. 执行 HcclReduceScatterInner */
    ret = HcclReduceScatterInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    /* 3. 校验执行结果准确性 */
    s32 errors = 0;
    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }
    EXPECT_EQ(errors, 0);

    /* 4. 释放相关资源 */
    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ret = hrtResetDevice(deviceLogicID);
    EXPECT_EQ(ret, 0);
    return;
}
#if 0 // 执行失败Unknown comm devType
TEST_F(OpbaseMultiThreadTest, ut_HcclReduceScatter)
{
    int ret = HCCL_SUCCESS;
    uint32_t ndev = 1;
    int32_t devices[ndev] = {0};
    HcclComm comms[ndev];
    for (int i = 0; i < ndev; i++) {
        ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, 0);

    std::vector<std::thread> threads;
    threads.resize(ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i] = std::thread(ExecReduceScatter, ndev, comms[i], devices[i], i);
    }

    for (uint32_t i = 0; i < ndev; ++i) {
        threads[i].join();
    }

    for (uint32_t i = 0; i < ndev; i++) {
        ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
        ret = HcclCommDestroy(comms[i]);
        EXPECT_EQ(ret, 0);
    }
}
#endif
void ExecReduce(int ndev, HcclComm comm, uint32_t deviceLogicID, uint32_t rankId)
{
    int ret = HCCL_SUCCESS;
    /* 1. 申请相关资源 */
    ret = hrtSetDevice(deviceLogicID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtStream_t stream;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    s32 count = HCCL_COM_DATA_SIZE;

    s8* sendbuf;
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

    s8* recvbuf;
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    uint32_t rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, ndev);

    uint32_t rankID = 0;
    ret = HcclGetRankId(comm, &rankID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankID, rankId);

    /* 2. 执行 HcclReduceInner */
    ret = HcclReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    /* 3. 校验执行结果准确性 */

    s32 errors = 0;

    if (rankId == 0) {
        for (int j = 0; j < count; j++) {
            if (recvbuf[j] != 2) {
                printf("rankId : %d, deviceLogicID: %d, j : %d, val : %d \n", rankId, deviceLogicID, j, recvbuf[j]);
                errors++;
                break;
            }
        }
    }
    EXPECT_EQ(errors, 0);

    /* 4. 释放相关资源 */
    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ret = hrtResetDevice(deviceLogicID);
    EXPECT_EQ(ret, 0);
    return;
}
#if 0 // 执行失败Unknown comm devType
TEST_F(OpbaseMultiThreadTest, ut_HcclReduce)
{
    int ret = HCCL_SUCCESS;
    uint32_t ndev = 1;
    int32_t devices[ndev] = {0};
    HcclComm comms[ndev];
    for (int i = 0; i < ndev; i++) {
        ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, 0);

    std::vector<std::thread> threads;
    threads.resize(ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i] = std::thread(ExecReduce, ndev, comms[i], devices[i], i);
    }

    for (uint32_t i = 0; i < ndev; ++i) {
        threads[i].join();
    }

    for (uint32_t i = 0; i < ndev; i++) {
        ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
        ret = HcclCommDestroy(comms[i]);
        EXPECT_EQ(ret, 0);
    }
}
#endif

// 覆盖 op_base.RunGather 空 addrInfo 分支：清零 sendCounts/sdispls 后直接返回成功，避免越界访问
TEST(OpbaseRunGatherTest, ut_RunGatherEmptyAddrInfo)
{
    GatherPara gatherPara;
    gatherPara.rankSize = 4;
    gatherPara.addrLength = -1; // 退化输入下 addrInfo 为空数组
    u64 sendCounts[4] = {1, 1, 1, 1};
    u64 sdispls[4] = {0, 1, 2, 3};
    HcclResult ret = RunGather(sendCounts, sdispls, nullptr, gatherPara);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (u32 i = 0; i < gatherPara.rankSize; i++) {
        EXPECT_EQ(sendCounts[i], 0);
        EXPECT_EQ(sdispls[i], 0);
    }
    GlobalMockObject::verify();
}
