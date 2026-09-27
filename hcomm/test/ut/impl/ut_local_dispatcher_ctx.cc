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
#include <cmath>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

#include "workflow_pub.h"
#include "topoinfo_struct.h"
#include "hccl_ip_address.h"
#include "dlra_function.h"
#include "hccl_net_dev.h"
#define private public
#define protected public
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_aiv.h"
#include "config.h"
#include "externalinput.h"
#include "hccl_communicator.h"
#include "hccl_communicator_attrs.h"
#include "hccl_comm_pub.h"
#include "transport_base_pub.h"
#include "comm_impl.h"
#include "comm_mesh_pub.h"
#include "coll_alg_operator.h"
#include "all_gather_operator.h"
#include "reduce_operator.h"
#include "transport_pub.h"
#include "hccl_common.h"
#include "broadcast_operator.h"
#include "reduce_scatter_operator.h"
#include "notify_pool.h"
#include "comm_base_pub.h"
#include "task_abort_handler_pub.h"
#include "coll_comm_executor.h"
#include "adapter_rts.h"
#include "heartbeat.h"
#include "acl/acl.h"
#include "hccl_comm.h"
#include "hccl_inner.h"
#undef private
#undef protected
#include "remote_notify.h"
#include "profiling_manager.h"
#include "base.h"
#include "adapter_rts_common.h"
#include "hdc_pub.h"
#include "aicpu_hdc_utils.h"
#include "common/aicpu_kfc_def.h"
#include "llt_hccl_stub_mc2.h"
#include "llt_hccl_stub.h"
#include "dispatcher_ctx.h"
#include "hccl_dispatcher_ctx.h"
#include "hccl_primitive_local.h"
#include "hccl_tbe_task.h"

using namespace std;

class LocalCtxUt : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

HcclResult hrtDrvGetPlatformInfoStub(uint32_t* info)
{
    *info = 1;
    return HCCL_SUCCESS;
}

TEST_F(LocalCtxUt, CreateCtxAiCpuMode)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, CreateCtxFFTSMode)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(false));

    MOCKER(GetWorkflowMode)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, CreateCtxNorMalMode)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().with(mockcpp::any()).will(returnValue(false));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(false));

    MOCKER(GetWorkflowMode)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, CreateCtxERRMode)
{
    u32 devicePhyId = INVALID_UINT;
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(devicePhyId))
        .will(returnValue(HCCL_SUCCESS));
    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, devicePhyId);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalCopyAICpu)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    u64* ptr = new u64(1);
    HcclBuf dst{ptr, 1, nullptr};
    HcclBuf src{ptr, 1, nullptr};
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    ret = HcclLocalCopy(streamtemp, &dst, &src);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete ptr;
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalCopyFfts)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    u64* ptr = new u64(1);
    HcclBuf dst{ptr, 1, nullptr};
    HcclBuf src{ptr, 1, nullptr};
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    ret = HcclLocalCopy(streamtemp, &dst, &src);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete ptr;
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalCopyNormal)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    u64* ptr = new u64(1);
    HcclBuf dst{ptr, 1, nullptr};
    HcclBuf src{ptr, 1, nullptr};
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    ret = HcclLocalCopy(streamtemp, &dst, &src);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete ptr;
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalCopyERR)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    u64* ptr = new u64(1);
    HcclBuf dst{ptr, 1, nullptr};
    HcclBuf src{ptr, 2, nullptr};
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    ret = HcclLocalCopy(streamtemp, &dst, &src);
    EXPECT_NE(HCCL_SUCCESS, ret);
    delete ptr;
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalCopyWithReduceNormal)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherPub dispatcher(0);
    MOCKER_CPP_VIRTUAL(
        dispatcher, &DispatcherPub::InlineReduceAsync,
        HcclResult(DispatcherPub::*)(
            void const*, unsigned long long, HcclDataType, HcclReduceOp, hccl::Stream&, void*, unsigned int,
            hccl::LinkType))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);
    u64* ptr = new u64(1);
    HcclBuf dst{ptr, 1, nullptr};
    HcclBuf src{ptr, 1, nullptr};
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    HcclReduceInfo reduceInfo = {HcclDataType::HCCL_DATA_TYPE_INT16, HcclReduceOp::HCCL_REDUCE_SUM};
    ret = HcclLocalCopyReduce(streamtemp, &dst, &src, reduceInfo);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete ptr;
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalCopyWithReduceERR)
{
    u64* ptr = new u64(1);
    HcclBuf dst{ptr, 1, nullptr};
    HcclBuf src{ptr, 1, nullptr};
    aclrtStream streamtemp = nullptr;
    HcclReduceInfo reduceInfo = {HcclDataType::HCCL_DATA_TYPE_INT16, HcclReduceOp::HCCL_REDUCE_SUM};
    HcclResult ret = HcclLocalCopyReduce(streamtemp, &dst, &src, reduceInfo);
    EXPECT_NE(HCCL_SUCCESS, ret);
    delete ptr;
}

TEST_F(LocalCtxUt, LocalLaunchTaskExtendNormal)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    std::vector<hccl::Stream> subStreams_temp;
    std::vector<aclrtStream> subStreams;
    for (auto& s : subStreams_temp) {
        subStreams.push_back(reinterpret_cast<aclrtStream>(&s));
    }

    ret = HcclLocalLaunchTaskExtend(streamtemp, subStreams);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalLaunchTaskExtendERR)
{
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    std::vector<hccl::Stream> subStreams_temp;
    std::vector<aclrtStream> subStreams;
    for (auto& s : subStreams_temp) {
        subStreams.push_back(reinterpret_cast<aclrtStream>(&s));
    }

    HcclResult ret = HcclLocalLaunchTaskExtend(streamtemp, subStreams);
    EXPECT_NE(HCCL_SUCCESS, ret);
}

TEST_F(LocalCtxUt, LocalInitTaskNormal)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();

    ret = HcclLocalInitTask(streamtemp, true, "test");
    EXPECT_EQ(HCCL_SUCCESS, ret);
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, LocalInitTaskERR)
{
    Stream* stream = nullptr;
    aclrtStream streamtemp = reinterpret_cast<aclrtStream>(stream);
    HcclResult ret = HcclLocalInitTask(streamtemp, true, "test");
    EXPECT_NE(HCCL_SUCCESS, ret);
}

TEST_F(LocalCtxUt, HcclLocalNotifyRecordNormal)
{
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(hrtDrvGetPlatformInfoStub));

    MOCKER(hrtDrvGetPlatformInfo).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    DispatcherCtxPtr ctx;
    HcclResult ret = CreateDispatcherCtx(&ctx, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DispatcherCtx* ctxPtr = static_cast<DispatcherCtx*>(ctx);
    EXPECT_NE(ctxPtr->GetDispatcher(), nullptr);
    EXPECT_NE(GetDispatcherCtx(), nullptr);

    DispatcherPub* dispatcher = static_cast<DispatcherPub*>(ctxPtr->GetDispatcher());
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    aclrtStream streamtemp = stream.ptr();
    std::shared_ptr<LocalNotify> localNotifyPtr = std::make_shared<LocalNotify>();
    void* notify = static_cast<void*>(localNotifyPtr.get());
    ret = HcclLocalNotifyRecord(streamtemp, notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DestroyDispatcherCtx(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, HcclLocalNotifyRecordERR)
{
    aclrtStream streamtemp = nullptr;
    std::shared_ptr<LocalNotify> localNotifyPtr = std::make_shared<LocalNotify>();
    void* notify = static_cast<void*>(localNotifyPtr.get());
    HcclResult ret = HcclLocalNotifyRecord(streamtemp, notify);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(LocalCtxUt, HcclLocalWaitERR)
{
    aclrtStream streamtemp = nullptr;
    std::shared_ptr<LocalNotify> localNotifyPtr = std::make_shared<LocalNotify>();
    void* notify = static_cast<void*>(localNotifyPtr.get());
    HcclResult ret = HcclLocalNotifyWait(streamtemp, notify, INVALID_UINT);
    EXPECT_NE(ret, HCCL_SUCCESS);
}
