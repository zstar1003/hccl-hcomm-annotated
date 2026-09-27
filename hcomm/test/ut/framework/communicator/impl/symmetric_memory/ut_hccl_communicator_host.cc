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
#include <securec.h>

#ifndef private
#define private public
#define protected public
#endif
#include "hccl_communicator.h"
#undef private
#undef protected
#include "llt_hccl_stub_pub.h"
#include "adapter_rts_common.h"

using namespace std;
using namespace hccl;

class HcclCommunicatorHostTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclCommunicatorHostTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclCommunicatorHostTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "HcclCommunicatorHostTest Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "HcclCommunicatorHostTest Test TearDown" << std::endl;
    }
};

TEST_F(HcclCommunicatorHostTest, Ut_InitSymmetricMemory_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910_93;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_NE(hcclCommunicator->symmetricMemory_, nullptr);
}

TEST_F(HcclCommunicatorHostTest, Ut_InitSymmetricMemory_When_StrideIsValid_Expect_ReturnsIsHCCL_SUCCESS)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclSymWinMaxMemSizePerRank = 8;
    CommConfig commConfig;
    commConfig.Load(&config);
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator(commConfig));
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910_93;
    auto ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_NE(hcclCommunicator->symmetricMemory_, nullptr);
}

TEST_F(HcclCommunicatorHostTest, Ut_InitSymmetricMemory_When_SuperPodNumGreaterThanOne_Expect_ReturnIsHCCL_SUCCESS)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->superPodNum_ = 2;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // assert not created
    EXPECT_EQ(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_EQ(hcclCommunicator->symmetricMemory_, nullptr);
}

TEST_F(HcclCommunicatorHostTest, Ut_IsSupportSymmetricMemory_When_Normal_Expect_ReturnIsTrue)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemory::FindSymmetricWindow).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910_93;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_NE(hcclCommunicator->symmetricMemory_, nullptr);
    // 配置满足前置条件
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    hcclCommunicator->deviceNumPerAggregation_ = 2;
    hcclCommunicator->multiModuleDiffDeviceNumMode_ = false;

    OpParam opParam;
    opParam.inputSymWindow = reinterpret_cast<void*>(0x1000);
    opParam.outputSymWindow = reinterpret_cast<void*>(0x2000);
    opParam.aicpuUnfoldMode = true;

    bool retBool = hcclCommunicator->IsSupportSymmetricMemory(HcclCMDType::HCCL_CMD_ALLGATHER, opParam);
    EXPECT_EQ(retBool, true);
    GlobalMockObject::verify();
}

TEST_F(HcclCommunicatorHostTest, Ut_IsSupportSymmetricMemory_When_AicpuUnfoldIsFalse_Expect_ReturnIsFalse)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910_93;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_NE(hcclCommunicator->symmetricMemory_, nullptr);
    OpParam opParam;
    opParam.aicpuUnfoldMode = false;
    bool retBool = hcclCommunicator->IsSupportSymmetricMemory(HcclCMDType::HCCL_CMD_ALLGATHER, opParam);
    EXPECT_EQ(retBool, false);
}

TEST_F(HcclCommunicatorHostTest, Ut_IsSupportSymmetricMemory_When_WorkFlowModeIsNotOpBase_Expect_ReturnIsFalse)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910_93;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_NE(hcclCommunicator->symmetricMemory_, nullptr);
    OpParam opParam;
    opParam.aicpuUnfoldMode = true;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    bool retBool = hcclCommunicator->IsSupportSymmetricMemory(HcclCMDType::HCCL_CMD_ALLGATHER, opParam);
    EXPECT_EQ(retBool, false);
}

TEST_F(HcclCommunicatorHostTest, Ut_IsSupportSymmetricMemory_When_deviceTypeIsNot910_93_Expect_ReturnIsFalse)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910B;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    OpParam opParam;
    opParam.aicpuUnfoldMode = true;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    bool retBool = hcclCommunicator->IsSupportSymmetricMemory(HcclCMDType::HCCL_CMD_ALLGATHER, opParam);
    EXPECT_EQ(retBool, false);
}

TEST_F(
    HcclCommunicatorHostTest, Ut_IsSupportSymmetricMemory_When_multiModuleDiffDeviceNumModeIsTrue_Expect_ReturnIsFalse)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910_93;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_NE(hcclCommunicator->symmetricMemory_, nullptr);
    OpParam opParam;
    opParam.aicpuUnfoldMode = true;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    hcclCommunicator->multiModuleDiffDeviceNumMode_ = true;
    bool retBool = hcclCommunicator->IsSupportSymmetricMemory(HcclCMDType::HCCL_CMD_ALLGATHER, opParam);
    EXPECT_EQ(retBool, false);
}

TEST_F(
    HcclCommunicatorHostTest,
    Ut_IsSupportSymmetricMemory_When_FindSymmetricWindowReturnIsHCCL_E_NOT_FOUND_Expect_ReturnIsFalse)
{
    MOCKER_CPP(&SymmetricMemory::FindSymmetricWindow).stubs().will(returnValue(HCCL_E_NOT_FOUND));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910_93;
    HcclResult ret = hcclCommunicator->InitSymmetricMemory();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcclCommunicator->symmetricMemoryAgent_, nullptr);
    EXPECT_NE(hcclCommunicator->symmetricMemory_, nullptr);
    OpParam opParam;
    opParam.aicpuUnfoldMode = true;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    hcclCommunicator->deviceNumPerAggregation_ = 2;
    hcclCommunicator->multiModuleDiffDeviceNumMode_ = false;
    bool retBool = hcclCommunicator->IsSupportSymmetricMemory(HcclCMDType::HCCL_CMD_ALLGATHER, opParam);
    EXPECT_EQ(retBool, false);
    GlobalMockObject::verify();
}

class TestHcclCommunicator {
public:
    HcclResult AicpuInitOpTilingDataBuf(
        const OpParam& opParam, const HcclCMDType& opType, const std::string& kernelName,
        const AicpuOpTiling opTilingInfo, u64 dynamicDataSize)
    {
        MOCKER_CPP(&HcclCommunicator::InitAndCheckAicpuOrderNotify).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::BuildHierarchicalAlgOption).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::UpdateOpIndex).stubs().will(returnValue(0));
        MOCKER_CPP(&HcclCommunicator::GetOrderLaunchMode).stubs().will(returnValue(0));
        MOCKER_CPP(&HcclCommunicator::AicpuInitOpTilingDataFromOpParam).stubs().will(returnValue(HCCL_SUCCESS));

        return hcclCommunicator->AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize);
    }

    HcclCommunicator* hcclCommunicator;
};

TEST_F(HcclCommunicatorHostTest, Ut_SetDynamicTilingData_When_A2GroupSendRecv_Expect_SkipIsDirectRemoteRank)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->rankInfoList_.resize(2);
    hcclCommunicator->realUserRank_ = 0;
    hcclCommunicator->deviceType_ = DevType::DEV_TYPE_910B;
    hcclCommunicator->isGroupMode_ = true;
    hcclCommunicator->userRankSize_ = 2;

    OpParam opParam;
    opParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    opParam.BatchSendRecvDataDes.itemNum = 1;

    HcclSendRecvItem sendRecvInfo;
    opParam.BatchSendRecvDataDes.sendRecvItemsPtr = &sendRecvInfo;

    u8 isDirectRemoteRank[2] = {1, 0};
    opParam.BatchSendRecvDataDes.isDirectRemoteRank = isDirectRemoteRank;

    AicpuOpTiling opTilingInfo;
    opTilingInfo.algName = "test_alg";
    opTilingInfo.newTag = "test_new_tag";
    opTilingInfo.floatOverflowMode = 0;
    opTilingInfo.dumpDebug = 0;

    std::string kernelName = "test_kernel";
    u64 dynamicDataSize = 0;

    TestHcclCommunicator testComm;
    testComm.hcclCommunicator = hcclCommunicator.get();

    HcclResult ret = testComm.AicpuInitOpTilingDataBuf(
        opParam, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, kernelName, opTilingInfo, dynamicDataSize);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(hcclCommunicator->deviceType_, DevType::DEV_TYPE_910B);
    EXPECT_EQ(hcclCommunicator->isGroupMode_, true);
    EXPECT_EQ(hcclCommunicator->userRankSize_, 2);
}

static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 4;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910_93;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(4);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1694542017);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 0;
    rankVec[1].serverId = "192.168.0.101";
    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1711319232);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
    rankVec[2].serverIdx = 1;
    rankVec[2].serverId = "192.168.0.102";
    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr4(1711319233);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
    rankVec[3].serverIdx = 1;
    rankVec[3].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 4;
    rankTable.deviceNum = 4;
    rankTable.serverNum = 2;
}

TEST_F(HcclCommunicatorHostTest, Ut_HcclGetAlgExecParam_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    HcclResult ret = HCCL_SUCCESS;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::AllocAlgResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitProfiler).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    std::string tag = "test";
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    u64 count = 1024;

    void* inputPtr = malloc(count * sizeof(int8_t));
    void* outputPtr = malloc(count * sizeof(int8_t));

    bool clearEnable = true;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    void* commContext = nullptr;
    u64 len = 0;
    u32 aivCoreLimit = 2;

    ret = implBase->HcclGetAlgExecParam(
        tag, opType, count, inputPtr, outputPtr, clearEnable, dataType, op, commContext, len, aivCoreLimit);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    free(inputPtr);
    free(outputPtr);
}
TEST_F(HcclCommunicatorHostTest, Ut_CopyHostListResToDeviceParam_FirstTime_EmptyList_Expect_Success)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    // 空链表：sentinel 指向自身，while 条件立即为 false
    ListCommon sentinel;
    sentinel.nextHost = reinterpret_cast<u64>(&sentinel);
    sentinel.preHost = reinterpret_cast<u64>(&sentinel);
    sentinel.nextDevice = reinterpret_cast<u64>(&sentinel);
    sentinel.preDevice = reinterpret_cast<u64>(&sentinel);

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    std::string newTag = "test_tag";
    HcclResult ret = hcclCommunicator->CopyHostListResToDeviceParam(newTag, &sentinel, sizeof(HccltagLocalResV2));

    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclCommunicatorHostTest, Ut_CopyHostListResToDeviceParam_FirstTime_MultiNode_Expect_CopyFirstTwoOnly)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    // 构建4个节点的循环链表：sentinel → n1 → n2 → n3 → sentinel
    ListCommon sentinel;
    HccltagLocalResV2 n1, n2, n3;
    memset(&n1, 0, sizeof(n1));
    memset(&n2, 0, sizeof(n2));
    memset(&n3, 0, sizeof(n3));

    // host 链
    sentinel.nextHost = reinterpret_cast<u64>(&n1.nextTagRes);
    n1.nextTagRes.nextHost = reinterpret_cast<u64>(&n2.nextTagRes);
    n2.nextTagRes.nextHost = reinterpret_cast<u64>(&n3.nextTagRes);
    n3.nextTagRes.nextHost = reinterpret_cast<u64>(&sentinel);

    // device 链（只填充 host 节点的 nextDevice，供 memcpy dst 用；device 节点本身不遍历）
    ListCommon dev1, dev2, dev3;
    sentinel.nextDevice = reinterpret_cast<u64>(&dev1);
    n1.nextTagRes.nextDevice = reinterpret_cast<u64>(&dev2);
    n2.nextTagRes.nextDevice = reinterpret_cast<u64>(&dev3);
    n3.nextTagRes.nextDevice = reinterpret_cast<u64>(&sentinel);

    // 首次分配（newTagResAlloced_ 为空），应只拷前2个节点
    MOCKER(hrtMemSyncCopy).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));

    std::string newTag = "test_tag";
    HcclResult ret = hcclCommunicator->CopyHostListResToDeviceParam(newTag, &sentinel, sizeof(HccltagLocalResV2));

    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclCommunicatorHostTest, Ut_CopyHostListResToDeviceParam_RefreshSingleNode_TagFound_Expect_CopyOneOnly)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    // 预先插入 tag，使 isRefreshSingleNode = true
    std::string targetTag = "refresh_tag";
    hcclCommunicator->newTagResAlloced_.insert(targetTag);

    // 构建3个节点的链表，第二个节点匹配 targetTag
    ListCommon sentinel;
    HccltagLocalResV2 n1, n2, n3;
    memset(&n1, 0, sizeof(n1));
    memset(&n2, 0, sizeof(n2));
    memset(&n3, 0, sizeof(n3));
    memcpy_s(n1.tag, TAG_MAX_LENGTH, "other_tag", sizeof("other_tag"));
    memcpy_s(n2.tag, TAG_MAX_LENGTH, targetTag.c_str(), targetTag.length() + 1);
    memcpy_s(n3.tag, TAG_MAX_LENGTH, "another_tag", sizeof("another_tag"));

    sentinel.nextHost = reinterpret_cast<u64>(&n1.nextTagRes);
    n1.nextTagRes.nextHost = reinterpret_cast<u64>(&n2.nextTagRes);
    n2.nextTagRes.nextHost = reinterpret_cast<u64>(&n3.nextTagRes);
    n3.nextTagRes.nextHost = reinterpret_cast<u64>(&sentinel);

    ListCommon dev1, dev2, dev3;
    sentinel.nextDevice = reinterpret_cast<u64>(&dev1);
    n1.nextTagRes.nextDevice = reinterpret_cast<u64>(&dev2);
    n2.nextTagRes.nextDevice = reinterpret_cast<u64>(&dev3);
    n3.nextTagRes.nextDevice = reinterpret_cast<u64>(&sentinel);

    // 匹配到第二个节点时 break，只拷1次
    MOCKER(hrtMemSyncCopy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcclCommunicator->CopyHostListResToDeviceParam(targetTag, &sentinel, sizeof(HccltagLocalResV2));

    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclCommunicatorHostTest, Ut_CopyHostListResToDeviceParam_RefreshSingleNode_TagNotFound_Expect_NoCopy)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    // 预先插入 tag，使 isRefreshSingleNode = true
    std::string targetTag = "refresh_tag";
    hcclCommunicator->newTagResAlloced_.insert(targetTag);

    // 构建3个节点的链表，没有节点匹配 targetTag
    ListCommon sentinel;
    HccltagLocalResV2 n1, n2, n3;
    memset(&n1, 0, sizeof(n1));
    memset(&n2, 0, sizeof(n2));
    memset(&n3, 0, sizeof(n3));
    memcpy_s(n1.tag, TAG_MAX_LENGTH, "tag_a", sizeof("tag_a"));
    memcpy_s(n2.tag, TAG_MAX_LENGTH, "tag_b", sizeof("tag_b"));
    memcpy_s(n3.tag, TAG_MAX_LENGTH, "tag_c", sizeof("tag_c"));

    sentinel.nextHost = reinterpret_cast<u64>(&n1.nextTagRes);
    n1.nextTagRes.nextHost = reinterpret_cast<u64>(&n2.nextTagRes);
    n2.nextTagRes.nextHost = reinterpret_cast<u64>(&n3.nextTagRes);
    n3.nextTagRes.nextHost = reinterpret_cast<u64>(&sentinel);

    ListCommon dev1, dev2, dev3;
    sentinel.nextDevice = reinterpret_cast<u64>(&dev1);
    n1.nextTagRes.nextDevice = reinterpret_cast<u64>(&dev2);
    n2.nextTagRes.nextDevice = reinterpret_cast<u64>(&dev3);
    n3.nextTagRes.nextDevice = reinterpret_cast<u64>(&sentinel);

    // 没有匹配tag，不应触发拷贝
    MOCKER(hrtMemSyncCopy).expects(atMost(0)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcclCommunicator->CopyHostListResToDeviceParam(targetTag, &sentinel, sizeof(HccltagLocalResV2));

    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(
    HcclCommunicatorHostTest, Ut_CopyHostListResToDeviceParam_RefreshSingleNode_RemoteRes_TagFound_Expect_CopyOneOnly)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    std::string targetTag = "refresh_tag";
    hcclCommunicator->newTagResAlloced_.insert(targetTag);

    // 使用 HccltagRemoteResV2 类型（size == sizeof(HccltagRemoteResV2) 分支）
    ListCommon sentinel;
    HccltagRemoteResV2 n1, n2;
    memset(&n1, 0, sizeof(n1));
    memset(&n2, 0, sizeof(n2));
    memcpy_s(n1.tag, TAG_MAX_LENGTH, "other_tag", sizeof("other_tag"));
    memcpy_s(n2.tag, TAG_MAX_LENGTH, targetTag.c_str(), targetTag.length() + 1);

    sentinel.nextHost = reinterpret_cast<u64>(&n1.nextTagRes);
    n1.nextTagRes.nextHost = reinterpret_cast<u64>(&n2.nextTagRes);
    n2.nextTagRes.nextHost = reinterpret_cast<u64>(&sentinel);

    ListCommon dev1, dev2;
    sentinel.nextDevice = reinterpret_cast<u64>(&dev1);
    n1.nextTagRes.nextDevice = reinterpret_cast<u64>(&dev2);
    n2.nextTagRes.nextDevice = reinterpret_cast<u64>(&sentinel);

    // HccltagRemoteResV2 路径，匹配到第二个节点后 break
    MOCKER(hrtMemSyncCopy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcclCommunicator->CopyHostListResToDeviceParam(targetTag, &sentinel, sizeof(HccltagRemoteResV2));

    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(
    HcclCommunicatorHostTest,
    Ut_AicpuInitOpTilingDataFromOpParam_When_AicpuUnfoldModeEnabled_Expect_PropagatedToTilingData)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    ASSERT_NE(hcclCommunicator, nullptr);

    OpParam opParam;
    opParam.aicpuUnfoldMode = true;
    struct OpTilingData tilingData;
    memset_s(&tilingData, sizeof(tilingData), 0, sizeof(tilingData));

    HcclResult ret = hcclCommunicator->AicpuInitOpTilingDataFromOpParam(
        opParam, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, &tilingData);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tilingData.aicpuUnfoldMode, static_cast<u8>(opParam.aicpuUnfoldMode));
    EXPECT_EQ(tilingData.aicpuUnfoldMode, 1U);
}

TEST_F(
    HcclCommunicatorHostTest,
    Ut_AicpuInitOpTilingDataFromOpParam_When_AicpuUnfoldModeDisabled_Expect_PropagatedToTilingData)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    ASSERT_NE(hcclCommunicator, nullptr);

    OpParam opParam;
    opParam.aicpuUnfoldMode = false;
    struct OpTilingData tilingData;
    memset_s(&tilingData, sizeof(tilingData), 0, sizeof(tilingData));

    HcclResult ret = hcclCommunicator->AicpuInitOpTilingDataFromOpParam(
        opParam, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, &tilingData);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tilingData.aicpuUnfoldMode, static_cast<u8>(opParam.aicpuUnfoldMode));
    EXPECT_EQ(tilingData.aicpuUnfoldMode, 0U);
}
