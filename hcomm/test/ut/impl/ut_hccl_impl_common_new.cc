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

#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <cassert>
#include <semaphore.h>
#include <csignal>
#include <syscall.h>
#include <sys/prctl.h>
#include <syslog.h>
#include <unistd.h>
#include <cerrno>

#include <securec.h>

#include <sys/types.h>
#include <cstddef>
#include <sys/mman.h>
#include <fcntl.h>
#include <driver/ascend_hal.h>

#define private public
#define protected public

#include "opexecounter_pub.h"
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "hccl_communicator.h"
#include <sys/mman.h>
#include <fcntl.h>
#include "sal.h"
#include "dlra_function.h"
#include "externalinput_pub.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"
#include "rank_consistentcy_checker.h"
#include "workflow_pub.h"
#include "workflow_pub.h"
#include "dltdt_function.h"
#include "param_check_pub.h"
#include "externalinput.h"
#include <iostream>
#include <fstream>
#include <stream_utils.h>

#undef protected
#undef private

using namespace std;
using namespace hccl;

class HcclImplCommonNewTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclImplCommonNewTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclImplCommonNewTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_postSync)
{
    u64 one = 1;
    MOCKER_CPP(&HcclCommunicator::CalcOpTilingDynamicDataSize).stubs().will(returnValue(one));
    MOCKER_CPP(&HcclCommunicator::AicpuInitOpTilingDataBuf).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::AicpuKfcTilingDataLaunchIn).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    string commId = "AllReduce";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    aclrtSetDevice(0);

    HcclRtStream opStream;
    rtStream_t stream;
    HcclCommunicator communicator;
    HcclResult ret = communicator.Init(params, rankTable);

    bool bret = communicator.GetCommResource(" ", nullptr);
    EXPECT_EQ(bret, false);

    OpParam opParam;
    communicator.retryEnable_ = true;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    DeviceMem deviceContext;
    std::string kernelName = "";
    AicpuOpTiling opTilingInfo;
    ret = communicator.AicpuKfcTilingDataLaunchExt(opParam, opType, deviceContext, kernelName, opTilingInfo);
}

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_AicpuKfcTilingDataLaunchExt_Reduce_PostSync)
{
    u64 one = 1;
    MOCKER_CPP(&HcclCommunicator::CalcOpTilingDynamicDataSize).stubs().will(returnValue(one));
    MOCKER_CPP(&HcclCommunicator::AicpuInitOpTilingDataBuf).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::AicpuKfcTilingDataLaunchIn).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    string commId = "Reduce";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    aclrtSetDevice(0);

    HcclRtStream opStream;
    rtStream_t stream;
    HcclCommunicator communicator;
    HcclResult ret = communicator.Init(params, rankTable);
    communicator.superPodNum_ = 2;
    bool bret = communicator.GetCommResource(" ", nullptr);
    EXPECT_EQ(bret, false);

    OpParam opParam;
    communicator.retryEnable_ = true;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE;
    DeviceMem deviceContext;
    std::string kernelName = "";
    AicpuOpTiling opTilingInfo;
    ret = communicator.AicpuKfcTilingDataLaunchExt(opParam, opType, deviceContext, kernelName, opTilingInfo);
}

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_AicpuKfcTilingDataLaunchExt_ReduceScatter_PostSync)
{
    u64 one = 1;
    MOCKER_CPP(&HcclCommunicator::CalcOpTilingDynamicDataSize).stubs().will(returnValue(one));
    MOCKER_CPP(&HcclCommunicator::AicpuInitOpTilingDataBuf).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::AicpuKfcTilingDataLaunchIn).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    string commId = "ReduceScatter";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    aclrtSetDevice(0);

    HcclRtStream opStream;
    rtStream_t stream;
    HcclCommunicator communicator;
    HcclResult ret = communicator.Init(params, rankTable);
    communicator.superPodNum_ = 2;

    bool bret = communicator.GetCommResource(" ", nullptr);
    EXPECT_EQ(bret, false);

    OpParam opParam;
    communicator.retryEnable_ = true;
    communicator.inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::USER_LARGER_THAN_CCL;

    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    DeviceMem deviceContext;
    std::string kernelName = "";
    AicpuOpTiling opTilingInfo;
    ret = communicator.AicpuKfcTilingDataLaunchExt(opParam, opType, deviceContext, kernelName, opTilingInfo);
}

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_AicpuKfcTilingDataLaunchExt_Allreduce_PreSync)
{
    u64 one = 1;
    MOCKER_CPP(&HcclCommunicator::CalcOpTilingDynamicDataSize).stubs().will(returnValue(one));
    MOCKER_CPP(&HcclCommunicator::AicpuInitOpTilingDataBuf).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::AicpuKfcTilingDataLaunchIn).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    string commId = "Allreduce";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    aclrtSetDevice(0);

    HcclRtStream opStream;
    rtStream_t stream;
    HcclCommunicator communicator;
    HcclResult ret = communicator.Init(params, rankTable);
    communicator.superPodNum_ = 2;

    bool bret = communicator.GetCommResource(" ", nullptr);
    EXPECT_EQ(bret, false);

    OpParam opParam;
    communicator.retryEnable_ = true;
    communicator.inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::USER_LARGER_THAN_CCL;

    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    DeviceMem deviceContext;
    std::string kernelName = "";
    AicpuOpTiling opTilingInfo;
    ret = communicator.AicpuKfcTilingDataLaunchExt(opParam, opType, deviceContext, kernelName, opTilingInfo);
}

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_AicpuKDataLaunch)
{
    GlobalMockObject::verify();

    MOCKER(LocalNotify::Post).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    string commId = "Allreduce";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    aclrtSetDevice(0);

    HcclRtStream opStream;
    rtStream_t stream;
    HcclCommunicator communicator;
    HcclResult ret = communicator.Init(params, rankTable);
    communicator.superPodNum_ = 2;

    MOCKER_CPP_VIRTUAL(communicator, &HcclCommunicator::AicpuUnfoldKernelLaunchV2)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    bool bret = communicator.GetCommResource(" ", nullptr);
    EXPECT_EQ(bret, false);

    OpParam opParam;
    communicator.retryEnable_ = true;
    communicator.inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::USER_LARGER_THAN_CCL;

    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    DeviceMem deviceContext;
    std::string kernelName = "";
    AicpuOpTiling opTilingInfo;

    ret = communicator.AicpuKfcTilingDataLaunchIn(
        opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData));
    GlobalMockObject::verify();
}

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_AicpuKDataLaunch_Capture)
{
    GlobalMockObject::verify();

    MOCKER(LocalNotify::Post).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));

    MOCKER(rtStreamAddToModel).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclCommParams params;
    string commId = "Allreduce";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    aclrtSetDevice(0);

    HcclCommunicator communicator;
    HcclResult ret = communicator.Init(params, rankTable);
    communicator.superPodNum_ = 2;

    MOCKER_CPP_VIRTUAL(communicator, &HcclCommunicator::AicpuUnfoldKernelLaunchV2)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    bool bret = communicator.GetCommResource(" ", nullptr);
    EXPECT_EQ(bret, false);

    OpParam opParam;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.isCapture = true;
    communicator.retryEnable_ = false;
    communicator.inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::USER_LARGER_THAN_CCL;
    communicator.opStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    DeviceMem deviceContext;
    std::string kernelName = "";
    AicpuOpTiling opTilingInfo;

    uint64_t streamMode = 0;
    rtStream_t aicpuStream;
    ret = communicator.Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 重复调用
    ret = communicator.Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = communicator.AicpuKfcTilingDataLaunchIn(
        opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData));
    GlobalMockObject::verify();
}

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_GetStreamCaptureInfo)
{
    GlobalMockObject::verify();
    // 非单算子场景
    MOCKER(GetWorkflowMode)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));

    bool isCapture = false;
    aclmdlRI rtModel = nullptr;
    rtStream_t stream;
    HcclResult ret = GetStreamCaptureInfo(stream, rtModel, isCapture);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    // unsupport场景
    MOCKER(GetWorkflowMode)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(ACL_ERROR_RT_FEATURE_NOT_SUPPORT));
    ret = GetStreamCaptureInfo(stream, rtModel, isCapture);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    // capture异常场景
    MOCKER(GetWorkflowMode)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));
    ret = GetStreamCaptureInfo(stream, rtModel, isCapture);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));
    ret = GetStreamCaptureInfo(stream, rtModel, isCapture);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplCommonNewTest, ut_HcclCommunicator_AddStreamToModel)
{
    MOCKER(rtStreamAddToModel).stubs().with(mockcpp::any()).will(returnValue(1));
    rtModel_t rtModel = nullptr;
    rtStream_t stream;
    HcclResult ret = AddStreamToModel(rtModel, stream);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}

TEST_F(HcclImplCommonNewTest, ut_AicpuInitOpTilingDataAicpuCache_When_AclgraphCacheDisable_Expect_NoIncrement)
{
    HcclCommunicator communicator;
    OpParam opParam;
    opParam.aicpuCacheEnable = 0;
    opParam.isCapture = true;
    struct OpTilingData opTilingData;
    memset_s(&opTilingData, sizeof(OpTilingData), 0, sizeof(OpTilingData));

    HcclResult ret
        = communicator.AicpuInitOpTilingDataAicpuCache(opParam, HcclCMDType::HCCL_CMD_ALLREDUCE, &opTilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(opTilingData.aicpuCacheEnable, 0);
}

TEST_F(HcclImplCommonNewTest, ut_AicpuInitOpTilingDataAicpuCache_When_AclgraphCacheEnable_Expect_NoIncrement)
{
    HcclCommunicator communicator;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    OpParam opParam;
    opParam.aicpuCacheEnable = 1;
    opParam.isCapture = true;
    struct OpTilingData opTilingData;
    memset_s(&opTilingData, sizeof(OpTilingData), 0, sizeof(OpTilingData));

    HcclResult ret
        = communicator.AicpuInitOpTilingDataAicpuCache(opParam, HcclCMDType::HCCL_CMD_ALLREDUCE, &opTilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(opTilingData.aicpuCacheEnable, 1);
}

TEST_F(HcclImplCommonNewTest, ut_AicpuInitOpTilingDataAicpuCache_When_AclgraphZerocopyCacheEnable_Expect_IncrementBy10)
{
    HcclCommunicator communicator;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    OpParam opParam;
    opParam.aicpuCacheEnable = 1;
    opParam.isCapture = true;
    struct OpTilingData opTilingData;
    memset_s(&opTilingData, sizeof(OpTilingData), 0, sizeof(OpTilingData));

    HcclResult ret
        = communicator.AicpuInitOpTilingDataAicpuCache(opParam, HcclCMDType::HCCL_CMD_ALLREDUCE, &opTilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(opTilingData.aicpuCacheEnable, 11);
}
