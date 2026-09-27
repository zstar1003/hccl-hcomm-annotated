/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif
#include "hccl_communicator.h"
#include "ccl_buffer_manager.h"
#include "dispatcher.h"
#include "hccl_ip_address.h"
#include "externalinput_pub.h"
#include "common.h"
#include "coll_alg_param.h"
#include "device_capacity.h"
#include "transport_pub.h"
#include "rank_consistentcy_checker.h"
#include "aicpu_operator_pub.h"
#undef private
#undef protected
#include "llt_hccl_stub_pub.h"

using namespace hccl;

typedef HcclResult (HcclCommunicator::*GetAlgInfoWithCmdType)(
    const std::string&, const std::string&, HcclCMDType, std::string&, std::string&);

static HcclResult MockGetAlgInfo_AivDirect(
    HcclCommunicator*, const std::string&, const std::string&, HcclCMDType, std::string& algName, std::string& newTag)
{
    algName = "RunAlltoAllAivDirect";
    newTag = "test_tag_RunAlltoAllAivDirect_device";
    return HCCL_SUCCESS;
}

static HcclResult MockGetAlgInfo_OtherAlg(
    HcclCommunicator*, const std::string&, const std::string&, HcclCMDType, std::string& algName, std::string& newTag)
{
    algName = "RunAllReduceRing";
    newTag = "test_tag_RunAllReduceRing_device";
    return HCCL_SUCCESS;
}

class HcclTransportResTest : public testing::Test {
protected:
    void SetUp() override
    {
        RankInfo rankInfo;
        rankInfo.userRank = 0;
        rankInfo.worldRank = 0;
        rankInfo.devicePhyId = 0;
        rankInfo.deviceType = DevType::DEV_TYPE_910B;
        rankInfo.nicIp.push_back(HcclIpAddress());
        rankInfo.backupNicIp.push_back(HcclIpAddress());
        rankInfo.hostPort = 0;
        rankInfo.serverId = "test_server";
        rankInfoList_.push_back(rankInfo);

        cclBufferManager_ = new CCLBufferManager();
        socketManager_ = new std::unique_ptr<HcclSocketManager>(nullptr);
        notifyPool_ = new std::unique_ptr<NotifyPool>(nullptr);

        transportManager_ = new TransportManager(
            *cclBufferManager_, *socketManager_, nullptr, *notifyPool_, rankInfoList_, 0, "test_identifier", 0,
            NICDeployment::NIC_DEPLOYMENT_DEVICE, false, false, false, nicRanksPort_, vnicRanksPort_, false, devIpAddr_,
            hostIp_, localVnicIp_, netDevCtxMap_);
    }

    void TearDown() override
    {
        delete transportManager_;
        transportManager_ = nullptr;
        delete socketManager_;
        socketManager_ = nullptr;
        delete notifyPool_;
        notifyPool_ = nullptr;
        delete cclBufferManager_;
        cclBufferManager_ = nullptr;
        GlobalMockObject::verify();
    }

    CCLBufferManager* cclBufferManager_ = nullptr;
    std::unique_ptr<HcclSocketManager>* socketManager_ = nullptr;
    std::unique_ptr<NotifyPool>* notifyPool_ = nullptr;
    TransportManager* transportManager_ = nullptr;
    std::vector<RankInfo> rankInfoList_;
    std::vector<u32> nicRanksPort_;
    std::vector<u32> vnicRanksPort_;
    std::vector<HcclIpAddress> devIpAddr_;
    HcclIpAddress hostIp_;
    HcclIpAddress localVnicIp_;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;
};

TEST_F(HcclTransportResTest, ut_SetMachinePara_When_isNpuDirectRoceTrue_Expect_qpModeIsNormal)
{
    MachinePara machinePara;
    RankInfo localRank;
    RankInfo remoteRank;

    MOCKER_CPP(&RankConsistentcyChecker::GetCheckFrame).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(IsSupportAtomicWrite).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transportManager_->SetMachinePara(
        "test_tag", MachineType::MACHINE_SERVER_TYPE, rankInfoList_[0].serverId, 0, true, LinkMode::LINK_RESERVED_MODE,
        {}, DeviceMem(), DeviceMem(), DeviceMem(), false, false, false, 0, 0, 0, machinePara, localRank, remoteRank,
        HcclNetDevCtx(), TransportLinkType::RESERVED, IndOpMem(), false, HcclCMDType::HCCL_CMD_INVALID, true);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(machinePara.qpMode, QPMode::NORMAL);
}

TEST_F(HcclTransportResTest, ut_SetMachinePara_When_isNpuDirectRoceFalse_Expect_qpModeNotNormal)
{
    MachinePara machinePara;
    RankInfo localRank;
    RankInfo remoteRank;

    MOCKER_CPP(&RankConsistentcyChecker::GetCheckFrame).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(IsSupportAtomicWrite).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = transportManager_->SetMachinePara(
        "test_tag", MachineType::MACHINE_SERVER_TYPE, rankInfoList_[0].serverId, 0, true, LinkMode::LINK_RESERVED_MODE,
        {}, DeviceMem(), DeviceMem(), DeviceMem(), false, false, false, 0, 0, 0, machinePara, localRank, remoteRank,
        HcclNetDevCtx(), TransportLinkType::RESERVED, IndOpMem(), false, HcclCMDType::HCCL_CMD_INVALID, false);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(machinePara.qpMode, QPMode::NORMAL);
}

TEST_F(HcclTransportResTest, ut_AllocComResourceByTiling_When_AlgNameIsRunAlltoAllAivDirect_Expect_isNpuDirectRoceTrue)
{
    std::unique_ptr<HcclCommunicator> hcclComm(new (std::nothrow) HcclCommunicator());

    HcclCombinOpParam* combinOparaPtr = new HcclCombinOpParam();
    sal_memset(combinOparaPtr, sizeof(HcclCombinOpParam), 0, sizeof(HcclCombinOpParam));
    auto combinOparaHostMem = std::make_shared<HostMem>();
    combinOparaHostMem->ptr_ = combinOparaPtr;
    combinOparaHostMem->size_ = sizeof(HcclCombinOpParam);
    combinOparaHostMem->owner_ = false;
    combinOparaHostMem->isRtsMem_ = false;
    hcclComm->combinOparaMem_ = combinOparaHostMem;

    OpParam opParam;
    opParam.tag = "test_tag";
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALL;

    GetAlgInfoWithCmdType getAlgInfoPtr = &HcclCommunicator::GetAlgInfo;
    MOCKER_CPP(getAlgInfoPtr).stubs().will(invoke(MockGetAlgInfo_AivDirect));

    MOCKER_CPP(&HcclCommunicator::CreateAndGetAiCpuNotifyWithNotifyRes).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = hcclComm->AllocComResourceByTiling("AlltoAll=level1:hierarchy", &opParam);
    EXPECT_EQ(opParam.isNpuDirectRoce, true);

    delete combinOparaPtr;
}

TEST_F(
    HcclTransportResTest, ut_AllocComResourceByTiling_When_AlgNameIsNotRunAlltoAllAivDirect_Expect_isNpuDirectRoceFalse)
{
    std::unique_ptr<HcclCommunicator> hcclComm(new (std::nothrow) HcclCommunicator());

    HcclCombinOpParam* combinOparaPtr = new HcclCombinOpParam();
    sal_memset(combinOparaPtr, sizeof(HcclCombinOpParam), 0, sizeof(HcclCombinOpParam));
    auto combinOparaHostMem = std::make_shared<HostMem>();
    combinOparaHostMem->ptr_ = combinOparaPtr;
    combinOparaHostMem->size_ = sizeof(HcclCombinOpParam);
    combinOparaHostMem->owner_ = false;
    combinOparaHostMem->isRtsMem_ = false;
    hcclComm->combinOparaMem_ = combinOparaHostMem;

    OpParam opParam;
    opParam.tag = "test_tag";
    opParam.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    GetAlgInfoWithCmdType getAlgInfoPtr = &HcclCommunicator::GetAlgInfo;
    MOCKER_CPP(getAlgInfoPtr).stubs().will(invoke(MockGetAlgInfo_OtherAlg));

    MOCKER_CPP(&HcclCommunicator::CreateAndGetAiCpuNotifyWithNotifyRes).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = hcclComm->AllocComResourceByTiling("AllReduce=level0:ring", &opParam);
    EXPECT_EQ(opParam.isNpuDirectRoce, false);

    delete combinOparaPtr;
}
