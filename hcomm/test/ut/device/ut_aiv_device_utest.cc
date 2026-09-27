/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif
#include "hccl_aiv.h"
#include "coll_all_to_all_executor.h"
#include "hccl_comm_pub.h"
#include "plugin_runner.h"
#include "hccl_alg.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class Aiv_Device_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Aiv_Device_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "Aiv_Device_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(Aiv_Device_UT, AivDeviceTest)
{
    AivProfilingInfo aivProInfo;
    uint64_t beginTime = 0;
    SetAivProfilingInfoBeginTime(beginTime);
    SetAivProfilingInfoBeginTime(aivProInfo);
}

TEST_F(Aiv_Device_UT, AlltoallExecutorDeviceTest)
{
    MOCKER_CPP(&CollAlltoAllExecutor::SetParallelTaskLoader).stubs().will(returnValue(HCCL_SUCCESS));
}

TEST_F(Aiv_Device_UT, CommDeviceTest)
{
    hcclComm comm;
    comm.RegistTaskAbortHandler();
    comm.UnRegistTaskAbortHandler();
    comm.GetOneSidedService(nullptr);
    comm.InitOneSidedServiceNetDevCtx(0);
    comm.DeinitOneSidedService();
}

TEST_F(Aiv_Device_UT, PluginRunnerDeviceTest)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
    PluginRunner pluginRunner(&taskExceptionHandler);
    rtStream_t stream;
    bool isCapture = true;
    pluginRunner.isStreamCapture(stream, isCapture);
}

TEST_F(Aiv_Device_UT, AlgDeviceTest)
{
    CCLBufferManager cclBufferManager;
    HcclDispatcher dispatcher = nullptr;
    HcclDispatcher vDispatcher = nullptr;
    HcclAlg* alg = new HcclAlg(cclBufferManager, dispatcher, vDispatcher);

    std::unique_ptr<WorkspaceResource> workSpaceRes = nullptr;
    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    HcclAlgoAttr algoAttr;
    HcclTopoAttr topoAttr;
    alg->Init(nullptr, 0, workSpaceRes, notifyPool, netDevCtxMap, queueNotifyManager, algoAttr, topoAttr, true);
    alg->Init(algoAttr, topoAttr, true);

    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    u64 memSize = 0;
    alg->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    alg->GetAllReduceScratchSize(0, dataType, memSize);
    TopoType topoType = TopoType::TOPO_TYPE_COMMON;
    alg->GetTopoType(topoType);
    AlgType algType;
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_ALL;
    alg->SetAlgType(algType, cmdType);

    bool flag = true;
    alg->SupportDeterministicOptim(flag);
    alg->GetDeterministicConfig();
    alg->SetDeterministicConfig(0);
    alg->SetAivModeConfig(true);
    alg->GetAicpuUnfoldConfig();
    alg->SetAicpuUnfoldConfig(true);
    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
    alg->GetRankVecInfo(serverAndsuperPodToRank);
    std::vector<bool> isBridgeVector;
    alg->GetIsBridgeVector(isBridgeVector);

    std::vector<std::vector<std::vector<u32>>> commPlaneRanks;
    alg->GetCommPlaneRanks(commPlaneRanks);
    std::vector<std::vector<std::vector<RankInfo>>> commPlaneVector;
    alg->GetCommPlaneVector(commPlaneVector);
    std::vector<std::vector<std::vector<std::vector<u32>>>> commPlaneSubGroupVector;
    alg->GetCommPlaneSubGroupVector(commPlaneSubGroupVector);
    std::map<AHCConcOpType, TemplateType> ahcAlgOption;
    alg->GetAHCAlgOption(ahcAlgOption);

    std::unordered_map<u32, bool> isUsedRdmaMap;
    alg->GetIsUsedRdmaMap(isUsedRdmaMap);
    DeviceMem deviceMem;
    alg->GetTinyMem(deviceMem);

    HcclExternalEnable externalEnable;
    alg->InitExternalEnable(externalEnable);
    HcclTopoInfo topoInfo;
    alg->InitTopoInfo(topoInfo, topoAttr);
    HcclAlgoInfo algoInfo;
    alg->InitAlgoInfo(algoInfo, algoAttr);
#ifndef OPEN_HCCL_TEST // 以下测试方法需对pimpl_做初始化才能调用 hccl_alg.cc中的Init方法受此宏控制
    alg->ReleaseCommInfos();
    std::string tag = "alg";
    Stream stream;

    alg->ClearOpResource(tag);

    level1StreamInfo_t streamInfo;
    alg->CreateMutiStreamRes(tag, stream, streamInfo, algType, true);
    std::unique_ptr<CommInfo> commInfo = nullptr;
    alg->CreateComm(tag, deviceMem, deviceMem, algType, commInfo, 0, true, true);
    alg->CreateComm(tag, deviceMem, deviceMem, algType, 0, true);
    u32 status;
    alg->Break();
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap;
    std::vector<u32> ranksPort;
    alg->SetHDCModeInfo(rankDevicePhyIdNicInfoMap, ranksPort, true, true);
#endif
    delete alg;
}
