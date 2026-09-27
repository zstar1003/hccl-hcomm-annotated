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
#define private public
#define protected public
#include "hvd_ops_kernel_info_store.h"
#include "ops_kernel_info_store_base.h"
#undef private
#undef protected
#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "hccl_comm_pub.h"
#include "sal.h"
#include "hccl_impl.h"
#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"

#include "plugin_manager.h"
#include "external/ge/ge_api_types.h"  // ge对内options
#include "framework/common/ge_types.h" // ge对外options
#include "hccl/hcom.h"
#include "hccl/hcom_executor.h"
#include "ranktable/v80_rank_table.h"
#include <iostream>
#include <fstream>
#include "graph/utils/node_utils.h"

#include <securec.h>
#include <functional>
#include "external/graph/tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/fmk_error_codes.h"
#include "external/graph/types.h"
#include "hccl/hcom.h"
#include "comm.h"
#include "workspace_mem.h"
#include "hvd_adapter.h"

using namespace std;
using namespace hccl;

class HvdOpsKernelInfoStoreTest : public testing::Test {
protected:
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

class NodeTest : public ge::Node {
public:
    NodeTest() { ; };
    ~NodeTest() { ; };
};

TEST_F(HvdOpsKernelInfoStoreTest, ut_GetSupportedOP)
{
    HvdOpsKernelInfoStore InfoStore;
    std::vector<std::string> hcclSupportOp;
    hcclSupportOp.push_back("HorovodBroadcast");
    hcclSupportOp.push_back("HorovodAllreduce");
    hcclSupportOp.push_back("HorovodAllgather");
    hcclSupportOp.push_back("HorovodWait");
    HcclResult ret = InfoStore.GetSupportedOP(hcclSupportOp);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_HCCLOpsKernel)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);
    HcclResult ret = InfoStore.HCCLOpsKernel(task, "aaaaaaa");
    EXPECT_EQ(ret, HCCL_E_PARA);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_HvdWaitOpKernel)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_WAIT;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);

    MOCKER_CPP(&HCCLOpsKernelInfoStore::GetStreamMainFromTaskInfo)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = InfoStore.HvdWaitOpKernel(task, "HorovodWait");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_LoadTask)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);

    MOCKER_CPP(&HvdOpsKernelInfoStore::HCCLOpsKernel).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ge::Status ret = InfoStore.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_UnloadTask)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);

    ge::Status ret = InfoStore.UnloadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_GetEvent)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);

    rtEvent_t newEvent;
    HcclResult ret = InfoStore.GetEvent(2, newEvent);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_DestroyEvent)
{
    HvdOpsKernelInfoStore InfoStore;
    HcclResult ret = InfoStore.DestroyEvent();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_GetDataTypeFromTaskInfo)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);

    HcclDataType DataType;
    HcclResult ret = InfoStore.GetDataTypeFromTaskInfo(task, DataType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
}

HcclResult fun3(void* data)
{
    unsigned long long* val = (unsigned long long*)data;
    HCCL_INFO("In function fun1, val is %lld", *val);
    return HCCL_SUCCESS;
}

HcclResult fun4(void* data)
{
    unsigned long long* val = (unsigned long long*)data;
    HCCL_INFO("In function fun2, val is %lld", *val * 2);
    return HCCL_SUCCESS;
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_HvdBroadcastOpKernel)
{
    HvdOpsKernelInfoStore InfoStore;
    ge::GETaskInfo task;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_BROADCAST;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);

    MOCKER(aclrtProcessReport).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchCallback).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(&HCCLOpsKernelInfoStore::GetStreamMainFromTaskInfo)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = InfoStore.HvdBroadcastOpKernel(task, "HorovodBroadcast");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* stream1;
    void* stream2;
    g_hvdAdapterGlobal.HvdAdapterInit(stream1, 0);
    g_hvdAdapterGlobal.HvdAdapterInit(stream2, 1);
    HcomRegHvdCallback(fun3);
    SaluSleep(500);
    HcomRegHvdCallback(fun4);
    SaluSleep(500);
    g_hvdAdapterGlobal.HvdAdapterDestroy();
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
    GlobalMockObject::verify();
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_HvdAllReduceOpKernel)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);
    MOCKER(aclrtProcessReport).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchCallback).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(&HCCLOpsKernelInfoStore::GetStreamMainFromTaskInfo)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = InfoStore.HvdAllReduceOpKernel(task, "HorovodAllreduce");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* stream1;
    void* stream2;
    g_hvdAdapterGlobal.HvdAdapterInit(stream1, 0);
    g_hvdAdapterGlobal.HvdAdapterInit(stream2, 1);
    HcomRegHvdCallback(fun3);
    SaluSleep(500);
    HcomRegHvdCallback(fun4);
    SaluSleep(500);
    g_hvdAdapterGlobal.HvdAdapterDestroy();
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
    GlobalMockObject::verify();
}

TEST_F(HvdOpsKernelInfoStoreTest, ut_HvdAllGatherOpKernel)
{
    HvdOpsKernelInfoStore InfoStore;
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(10 * sizeof(float));
    sal_memset(recv, 10 * sizeof(float), 0, 10 * sizeof(float));
    rtStream_t stream;
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskInfo task;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);
    task.kernelHcclInfo.resize(1);
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].input_name = "tensor1";
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HVD_KERNEL_OP_TYPE_ALLGATHER;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].dims.resize(1);
    MOCKER(aclrtProcessReport).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchCallback).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(&HCCLOpsKernelInfoStore::GetStreamMainFromTaskInfo)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = InfoStore.HvdAllGatherOpKernel(task, "HorovodAllgather");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* stream1;
    void* stream2;
    g_hvdAdapterGlobal.HvdAdapterInit(stream1, 0);
    g_hvdAdapterGlobal.HvdAdapterInit(stream2, 1);
    HcomRegHvdCallback(fun3);
    SaluSleep(500);
    HcomRegHvdCallback(fun4);
    SaluSleep(500);
    g_hvdAdapterGlobal.HvdAdapterDestroy();
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendbuf);
    sal_free(recv);
    GlobalMockObject::verify();
}
