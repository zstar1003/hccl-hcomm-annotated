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

#ifndef private
#define private public
#define protected public
#endif
#include "common/aicpu_kfc_def.h"
#include "framework/aicpu_kfc_rpc_server.h"
#include "framework/aicpu_hccl_process.h"
#include "framework/aicpu_hdc.h"
#include "framework/aicpu_kfc_process.h"
#include "framework/aicpu_kfc_retry_process.h"
#include "aicpu_kfc/aicpu_kfc_interface.h"
#include "algorithm/aicpu_allreduce.h"
#include "algorithm/task_orchestrator.h"
#include "algorithm/aicpu_dmy_cal_allreduce.h"
#include "coll_native_executor_base.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#include "dispatcher_aicpu.h"
#include "hccl_communicator.h"
#include "alg_template_base_pub.h"
#include "aicpu_kfc/common/aicpu_kfc_tiling_utils.h"
#include "framework/aicpu_kfc_rpc_serverv2.h"
#include "framework/aicpu_kfc_prof.h"
#include "common/aicpu_hccl_common.h"
#include "stream_pub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hccl_common.h"
#include "dispatcher_aicpu_pub.h"
#include "dtype_common.h"
#include "hdc_pub.h"
#include "dispatcher.h"
#include "adapter_rts.h"
#include "aicpu_kfc/common/aicpu_kfc_utils.h"
#include "hccl_mc2_ex.h"
#undef private
#undef protected

using namespace std;
using namespace HcclApi;
using namespace hccl;

constexpr uint32_t BEGINSQEPOS = 10;
constexpr uint32_t ENDSQEPOS = 100;
HcclResult QuerySqStatusByTypeStub1(uint32_t devId, uint32_t sqId, drvSqCqPropType_t type, uint32_t& outVal)
{
    outVal = ENDSQEPOS;
    return HCCL_SUCCESS;
}

void CallMC2MaintenanceThreadStub(AicpuComContext* ctx) { return; }

class AicpuUnfold_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuUnfold_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AicpuUnfold_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        set_board_id(0x0000);
        s32 ndev = 8;
        for (s32 i = 0; i < ndev; i++) {
            set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910B));
        }
        g_stubDevType = DevType::DEV_TYPE_910B;
        MOCKER(halGetDeviceInfo).stubs().with(mockcpp::any()).will(invoke(StubhalGetDeviceInfo));
        MOCKER(QuerySqStatusByType).stubs().with(mockcpp::any()).will(invoke(QuerySqStatusByTypeStub1));
        std::cout << "AicpuUnfold_UT Test SetUP" << std::endl;
        MOCKER(AicpuHcclProcess::CallMC2MaintenanceThread).stubs().will(invoke(CallMC2MaintenanceThreadStub));
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "AicpuUnfold_UT Test TearDown" << std::endl;
    }
};

#define TEST_RANK_SIZE 4
#define REMOTE_TAG_LOOP_NUM 3
string g_tag = "hcom_aicpu_unfold";

static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = TEST_RANK_SIZE;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(TEST_RANK_SIZE);

    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.101";

    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1694542017);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
    rankVec[2].serverIdx = 0;
    rankVec[2].serverId = "192.168.0.102";

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr4(1711319233);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
    rankVec[3].serverIdx = 1;
    rankVec[3].serverId = "192.168.0.102";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = TEST_RANK_SIZE;
    rankTable.serverNum = TEST_RANK_SIZE / 2;
}

template <typename T>
static HcclResult CopyVectorToDeviceMem(const u64 len, DeviceMem& dstDeviceMem, const std::vector<T>& srcVec)
{
#ifndef CCL_KERNEL_AICPU
    u64 memSize = len;
    dstDeviceMem = DeviceMem::alloc(memSize);
    CHK_RET(hrtMemSet(dstDeviceMem.ptr(), len, len));

    std::shared_ptr<HostMem> srcHostMem;
    HostMem tmpBuffer = HostMem::alloc(len);
    EXCEPTION_CATCH((srcHostMem = std::make_shared<HostMem>(std::move(tmpBuffer))), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(srcHostMem);
    CHK_SAFETY_FUNC_RET(memset_s(srcHostMem.get()->ptr(), len, 0, len));
    std::copy(srcVec.begin(), srcVec.end(), static_cast<T*>(srcHostMem.get()->ptr()));
    CHK_RET(hrtMemSyncCopy(
        dstDeviceMem.ptr(), len, srcHostMem.get()->ptr(), len, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
#endif
    return HCCL_SUCCESS;
}

static HcclResult BuildHierarchicalAlgOption(const std::string& algName, DeviceMem& dstTlvDeviceMem, u64& tlvLen)
{
#ifndef CCL_KERNEL_AICPU
    std::map<AHCConcOpType, TemplateType> hierarchicalAlgOption
        = {{{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_NB},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_NB},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_NB},

           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_RING},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_RING},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_RING},

           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_NB},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_NB},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_NB},

           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_RING},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_RING},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_RING}};

    if (dstTlvDeviceMem.ptr() == nullptr) {
        std::vector<u32> hierarchicalAlgOptionVec;
        for (auto it = hierarchicalAlgOption.begin(); it != hierarchicalAlgOption.end(); ++it) {
            HCCL_DEBUG(
                "[HcclCommunicator][BuildHierarchicalAlgOption] Level [%u], ConcType[%u] AHCOpType[%u], TemplateType "
                "[%u]",
                it->first.ahcLevel, it->first.concType, it->first.ahcOpType, it->second);
            hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->first.ahcLevel));
            hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->first.concType));
            hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->first.ahcOpType));
            hierarchicalAlgOptionVec.push_back(static_cast<u32>(it->second));
        }
        tlvLen = hierarchicalAlgOptionVec.size();
        CHK_RET(CopyVectorToDeviceMem(tlvLen * sizeof(u32), dstTlvDeviceMem, hierarchicalAlgOptionVec));
    }
#endif
    return HCCL_SUCCESS;
}

static HcclResult BuildOpTopoResVectorTlvParam(
    const std::string& algName, const std::vector<std::vector<std::vector<std::vector<u32>>>>& inputVectorInfo,
    DeviceMem& dstTlvDeviceMem, u64& tlvLen)
{
#ifndef CCL_KERNEL_AICPU
    vector<u32> tlv;
    CommonTlv commonTlv;
    for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
        for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
            for (u16 level2Idx = 0; level2Idx < inputVectorInfo[level0Idx][level1Idx].size(); level2Idx++) {
                commonTlv.type = (((level0Idx << TOP_HIERARCHICAL_COMM_LEVEL0_SHIFT) | level1Idx)
                                  << TOP_HIERARCHICAL_COMM_LEVEL1_SHIFT)
                                 | level2Idx;
                commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE))
                                   + inputVectorInfo[level0Idx][level1Idx][level2Idx].size() * sizeof(RANK_TYPE);
                tlv.push_back(commonTlv.type);
                tlv.push_back(commonTlv.length);
                tlv.insert(
                    tlv.end(), inputVectorInfo[level0Idx][level1Idx][level2Idx].begin(),
                    inputVectorInfo[level0Idx][level1Idx][level2Idx].end());
            }
        }
    }
    for (u64 idx = 0; idx < tlv.size(); idx++) {
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResVectorTlvParam] idx[%lu] tlv[%lu]", idx, tlv[idx]);
    }
    tlvLen = tlv.size() * sizeof(u32);
    CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
#endif
    return HCCL_SUCCESS;
}

static HcclResult BuildOpTopoResTlvParam(
    const std::string& algName, const std::vector<std::vector<std::vector<u32>>>& inputVectorInfo,
    DeviceMem& dstTlvDeviceMem, u64& tlvLen, bool errorcommonTlv = false)
{
#ifndef CCL_KERNEL_AICPU
    vector<u32> tlv;
    CommonTlv commonTlv;
    for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
        for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
            commonTlv.type = (level0Idx << TOP_COMM_LEVEL0_SHIFT | level1Idx);
            commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE))
                               + inputVectorInfo[level0Idx][level1Idx].size() * sizeof(RANK_TYPE) + errorcommonTlv;
            tlv.push_back(commonTlv.type);
            tlv.push_back(commonTlv.length);
            tlv.insert(
                tlv.end(), inputVectorInfo[level0Idx][level1Idx].begin(), inputVectorInfo[level0Idx][level1Idx].end());
        }
    }
    tlvLen = tlv.size() * sizeof(u32);
    CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
#endif
    return HCCL_SUCCESS;
}

static DeviceMem dstTlvDeviceMem;
static DeviceMem dstDeviceMem;
static DeviceMem nicListdstDeviceMem;
static DeviceMem serverAndsuperPodToRankDevice;
static DeviceMem IsUsedRdmaPairVecMem;
static DeviceMem pairLinkCounterVecMem;
static DeviceMem complanSubGroupRankDeviceMem;
static DeviceMem hierarchicalAlgOptionVecDeviceMem;
static std::vector<u32> niclist;
static std::vector<std::vector<std::vector<u32>>> commPlaneRanks
    = {{{0, 1}}, {}, {{0, 2}, {1, 3}}, {}, {{0}}, {{0, 1}}, {{0, 2}}, {{0, 1, 3, 2}}, {{0, 1, 2, 3}}};
static std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank
    = {{{0, 1}, {2, 3}},
       {
           {0, 1, 2, 3},
       }};

static std::vector<u32> isUsedRdmaPairVec = {3, 1, 2, 1, 1, 0, 0, 0};

static std::vector<u32> pairLinkCounterVec = {3, 0, 2, 0, 1, 0, 0, 2};
static std::vector<bool> bridgeRank{true, false};
static std::vector<std::vector<std::vector<std::vector<u32>>>> CommPlaneSubGroupVector = {{{{0, 1}, {2, 3}}}};

HcclResult TestConstructHcclOpResParam(
    HcclOpResParam& paramTask, std::vector<void*>& bufferVec, std::string& hcomId, SqCqeContext& sqeCqeContext)
{
    memset_s(&paramTask, sizeof(HcclOpResParam), 0, sizeof(HcclOpResParam));
    paramTask.localUsrRankId = 0;
    paramTask.rankSize = TEST_RANK_SIZE;
    strcpy(paramTask.hcomId, hcomId.c_str());
    paramTask.localRes.signalNum = LOCAL_NOTIFY_MAX_NUM;
    paramTask.localRes.streamNum = LOCAL_STREAM_MAX_NUM;
    for (uint32_t i = 0; i < LOCAL_STREAM_MAX_NUM; i++) {
        paramTask.localRes.streamParam[i].streamInfo.streamIds = 52 + i;
        paramTask.localRes.streamParam[i].streamInfo.sqIds = 52 + i;
        paramTask.localRes.streamParam[i].streamInfo.cqIds = 52 + i;
        paramTask.localRes.streamParam[i].streamInfo.logicCqids = 52 + i;
    }
    uint64_t resId = 0;
    for (uint32_t i = 0; i < LOCAL_NOTIFY_MAX_NUM; i++) {
        paramTask.localRes.localSignals[i].resId = ++resId;
        paramTask.localRes.localSignals[i].addr = 0x10 + i;
        paramTask.localRes.localSignals[i].devId = 0;
        paramTask.localRes.localSignals[i].tsId = 0;
        paramTask.localRes.localSignals[i].rankId = i;
    }
    DeviceMem aicpuOrderNotifyAddr_ = DeviceMem::alloc(sizeof(HcclSignalInfo) * AICPU_ORDER_NOTIFY_MAX_NUM);
    paramTask.aicpuOrderNotifyAddr = reinterpret_cast<u64>(aicpuOrderNotifyAddr_.ptr());
    HcclSignalInfo* signals = reinterpret_cast<HcclSignalInfo*>(aicpuOrderNotifyAddr_.ptr());
    for (u32 i = 0; i < AICPU_ORDER_NOTIFY_MAX_NUM; i++) {
        signals[i].resId = ++resId;
        signals[i].addr = 0x10000 + i * 0x100;
        signals[i].devId = 0;
        signals[i].tsId = 0;
        signals[i].rankId = 10 * i;
    }

    paramTask.localRes.mainStreamParam.streamInfo.streamIds = 1111;
    paramTask.localRes.mainStreamParam.streamInfo.sqIds = 1111;
    paramTask.localRes.mainStreamParam.streamInfo.cqIds = 1111;
    paramTask.localRes.mainStreamParam.streamInfo.logicCqids = 1111;

    paramTask.aicpuOrderStreamParam.streamInfo.streamIds = 1112;
    paramTask.aicpuOrderStreamParam.streamInfo.sqIds = 1112;
    paramTask.aicpuOrderStreamParam.streamInfo.cqIds = 1112;
    paramTask.aicpuOrderStreamParam.streamInfo.logicCqids = 1112;

    paramTask.localRes.mainStreamParam.sqCqContextAddr = reinterpret_cast<u64>(&sqeCqeContext);
    paramTask.localRes.mainStreamParam.sqCqContextSize = sizeof(SqCqeContext);
    for (u32 i = 0; i < paramTask.localRes.streamNum; i++) {
        paramTask.localRes.streamParam[i].sqCqContextAddr = reinterpret_cast<u64>(&sqeCqeContext);
        paramTask.localRes.streamParam[i].sqCqContextSize = sizeof(SqCqeContext);
    }
    paramTask.aicpuOrderStreamParam.sqCqContextAddr = reinterpret_cast<u64>(&sqeCqeContext);
    paramTask.aicpuOrderStreamParam.sqCqContextSize = sizeof(SqCqeContext);
    ListCommonInit(&paramTask.localRes.nextTagRes, &paramTask.localRes.nextTagRes);

    // 每张卡共有3个tag
    int loopNum = REMOTE_TAG_LOOP_NUM;
    while (loopNum > 0) {
        void* tmpTagBuffer = (void*)calloc(1, sizeof(HccltagLocalResV2));
        bufferVec.push_back(tmpTagBuffer);
        HccltagLocalResV2* tagBufferPtr = reinterpret_cast<HccltagLocalResV2*>(tmpTagBuffer);
        tagBufferPtr->ScratchmemSize = 0x1000000;
        tagBufferPtr->Scratchmem = 0x1000000;
        string newTag = g_tag + to_string(loopNum);
        strcpy(tagBufferPtr->tag, newTag.c_str());
        ListCommonInit(&tagBufferPtr->nextTagRes, &tagBufferPtr->nextTagRes);
        ListCommonAddHead(
            &tagBufferPtr->nextTagRes, &tagBufferPtr->nextTagRes, &paramTask.localRes.nextTagRes,
            &paramTask.localRes.nextTagRes);
        loopNum--;
    }

    // 共有16个卡
    paramTask.remoteResNum = TEST_RANK_SIZE;
    for (uint32_t i = 0; i < TEST_RANK_SIZE; i++) {
        void* tmpBuffer = (void*)calloc(1, sizeof(HcclRankRelationResV2));
        bufferVec.push_back(tmpBuffer);
        paramTask.remoteRes[i].nextDevicePtr = reinterpret_cast<u64>(tmpBuffer);
        HcclRankRelationResV2* bufferPtr = reinterpret_cast<HcclRankRelationResV2*>(tmpBuffer);
        if (bufferPtr == nullptr) {
            HCCL_ERROR("DDDDDDDDDDDDDD [%u]", sizeof(HcclRankRelationResV2));
            return HCCL_SUCCESS;
        }
        bufferPtr->windowsIn = i + 1;
        bufferPtr->windowsOut = i + 1;
        ListCommonInit(&bufferPtr->nextTagRes, &bufferPtr->nextTagRes);

        // 每张卡共有3个tag
        int loopNum = REMOTE_TAG_LOOP_NUM;

        while (loopNum > 0) {
            void* tmpTagBuffer = (void*)calloc(1, sizeof(HccltagRemoteResV2));
            bufferVec.push_back(tmpTagBuffer);
            HccltagRemoteResV2* tagBufferPtr = reinterpret_cast<HccltagRemoteResV2*>(tmpTagBuffer);
            string newTag = g_tag + to_string(loopNum);
            strcpy(tagBufferPtr->tag, newTag.c_str());

            for (uint32_t j = 0; j < LINK_P2P_MAX_NUM; j++) {
                tagBufferPtr->linkP2p.localIpcSignal[j].resId = ++resId;
                tagBufferPtr->linkP2p.localIpcSignal[j].addr = 0x2000 + j;
                tagBufferPtr->linkP2p.localIpcSignal[j].devId = 0;
                tagBufferPtr->linkP2p.localIpcSignal[j].tsId = 0;
                tagBufferPtr->linkP2p.localIpcSignal[j].rankId = j;

                tagBufferPtr->linkP2p.remoteIpcSignal[j].resId = ++resId;
                tagBufferPtr->linkP2p.remoteIpcSignal[j].addr = 0x2000 + j;
                tagBufferPtr->linkP2p.remoteIpcSignal[j].devId = 0;
                tagBufferPtr->linkP2p.remoteIpcSignal[j].tsId = 0;
                tagBufferPtr->linkP2p.remoteIpcSignal[j].rankId = j;

                tagBufferPtr->linkP2pSio.localIpcSignal[j].resId = ++resId;
                tagBufferPtr->linkP2pSio.localIpcSignal[j].addr = 0x2000 + j;
                tagBufferPtr->linkP2pSio.localIpcSignal[j].devId = 0;
                tagBufferPtr->linkP2pSio.localIpcSignal[j].tsId = 0;
                tagBufferPtr->linkP2pSio.localIpcSignal[j].rankId = j;

                tagBufferPtr->linkP2pSio.remoteIpcSignal[j].resId = ++resId;
                tagBufferPtr->linkP2pSio.remoteIpcSignal[j].addr = 0x2000 + j;
                tagBufferPtr->linkP2pSio.remoteIpcSignal[j].devId = 0;
                tagBufferPtr->linkP2pSio.remoteIpcSignal[j].tsId = 0;
                tagBufferPtr->linkP2pSio.remoteIpcSignal[j].rankId = j;
            }
            constexpr u32 RDMA_NOTIFY_MAX_NUM_UT = 8 * 3;
            std::vector<HcclSignalInfo> localInfo(RDMA_NOTIFY_MAX_NUM_UT);
            std::vector<u64> remoteInfo(RDMA_NOTIFY_MAX_NUM_UT);
            for (uint32_t k = 0; k < 1; k++) {
                tagBufferPtr->linkRoce[k].localMem[0].size = 4;
                tagBufferPtr->linkRoce[k].localMem[0].addr = 0x2000;
                tagBufferPtr->linkRoce[k].localMem[0].key = 0;
                tagBufferPtr->linkRoce[k].localMem[1].size = 4;
                tagBufferPtr->linkRoce[k].localMem[1].addr = 0x2001;
                tagBufferPtr->linkRoce[k].localMem[1].key = 0;
                tagBufferPtr->linkRoce[k].notifyValue = 0x4000;
                tagBufferPtr->linkRoce[k].notifyValueKey = 0;
                tagBufferPtr->linkRoce[k].remoteNotifyKey = 0;
                tagBufferPtr->linkRoce[k].chipId = 0;
                for (uint32_t j = 0; j < RDMA_NOTIFY_MAX_NUM_UT; j++) {
                    localInfo[j].resId = ++resId;
                    localInfo[j].addr = 0x3000 + j;
                    localInfo[j].devId = 0;
                    localInfo[j].tsId = 0;
                    localInfo[j].rankId = j;
                    remoteInfo[j] = 0x5000 + j;
                    tagBufferPtr->linkRoce[k].QpInfo[j / 3].qpPtr = 0x6000;
                    tagBufferPtr->linkRoce[k].QpInfo[j / 3].sqIndex = 1;
                    tagBufferPtr->linkRoce[k].QpInfo[j / 3].dbIndex = 2;
                }
                tagBufferPtr->linkRoce[k].localNotifyList = reinterpret_cast<u64>(localInfo.data());
                tagBufferPtr->linkRoce[k].remoteNotifyList = reinterpret_cast<u64>(remoteInfo.data());
            }

            ListCommonInit(&tagBufferPtr->nextTagRes, &tagBufferPtr->nextTagRes);
            ListCommonAddHead(
                &tagBufferPtr->nextTagRes, &tagBufferPtr->nextTagRes, &bufferPtr->nextTagRes, &bufferPtr->nextTagRes);
            loopNum--;
        }
    }

    {
        HcclResult ret = HCCL_SUCCESS;
        HcclCommParams params;
        RankTable_t rankTable;
        TestConstructParam(params, rankTable);
        params.deviceType = DevType::DEV_TYPE_910;

        std::string serverId;
        {
            for (u32 i = 0; i < rankTable.rankList.size(); i++) {
                if (rankTable.rankList[i].rankId == params.rank) {
                    serverId = rankTable.rankList[i].serverId;
                    break;
                }
            }
            if (serverId.empty()) {
                HCCL_ERROR("[Get][ServerId]GetServerId fail");
                return HCCL_E_PARA;
            }
            ServRankInfo_t servRankInfo;
            // 按server重新组织rank信息，便于后续校验及信息填写
            for (size_t index = 0; index < rankTable.rankList.size(); ++index) {
                const RankInfo_t& rankInfo = rankTable.rankList[index];
                std::string serverId = SalTrim(rankInfo.serverId);
                // 以serverID为索引，将server下的ranks放入vector
                ServRankInfo_t::iterator itr = servRankInfo.find(serverId);
                if (itr != servRankInfo.end()) {
                    itr->second.push_back(rankInfo);
                } else {
                    std::vector<RankInfo_t> rankInfoList;
                    rankInfoList.push_back(rankInfo);
                    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair(serverId, rankInfoList);
                    servRankInfo.insert(rankInfoPair);
                }
            }
            // 每个server下的rank列表按设备Id从小到大的顺序排序
            for (auto& iter : servRankInfo) {
                std::sort(
                    iter.second.begin(), iter.second.end(),
                    [&](const RankInfo_t& left, const RankInfo_t& right) -> bool {
                        return left.deviceInfo.devicePhyId < right.deviceInfo.devicePhyId;
                    });
            }
            niclist.clear();
            for (auto iter : servRankInfo[serverId]) {
                if (((!iter.hostIp.IsInvalid()) || (!iter.deviceInfo.deviceIp[0].IsInvalid()))
                    && (iter.deviceInfo.devicePhyId != HOST_DEVICE_ID)) {
                    niclist.push_back(iter.deviceInfo.devicePhyId);
                }
            }
            std::sort(niclist.begin(), niclist.end());
            EXPECT_EQ(2, niclist.size());
            u64 len = niclist.size() * sizeof(u32);
            CopyVectorToDeviceMem(len, nicListdstDeviceMem, niclist);
            paramTask.topoInfo.nicList = reinterpret_cast<u64>(nicListdstDeviceMem.ptr());
            paramTask.topoInfo.nicNum = niclist.size();
        }
        paramTask.topoInfo.userRank = params.rank;
        paramTask.topoInfo.userRankSize = params.totalRanks;
        paramTask.topoInfo.deviceLogicId = params.logicDevId;
        paramTask.topoInfo.isSingleMeshAggregation = false;
        paramTask.topoInfo.deviceNumPerAggregation = TEST_RANK_SIZE;
        paramTask.topoInfo.superPodNum = 0;
        paramTask.topoInfo.devicePhyId = 0;
        paramTask.topoInfo.deviceType = static_cast<u32>(params.deviceType);
        paramTask.topoInfo.topoType = 1;
        {
            u64 size = bridgeRank.size() * sizeof(bool);
            CopyVectorToDeviceMem(size, dstDeviceMem, bridgeRank);
            paramTask.topoInfo.bridgeRank = reinterpret_cast<u64>(dstDeviceMem.ptr());
            paramTask.topoInfo.bridgeRankNum = bridgeRank.size();
            string algName = "allreduce_mesh";
            u64 tlvLen = 0;
            CHK_RET(BuildOpTopoResTlvParam(algName, commPlaneRanks, dstTlvDeviceMem, tlvLen));
            paramTask.topoInfo.complanRank = reinterpret_cast<u64>(dstTlvDeviceMem.ptr());
            paramTask.topoInfo.complanRankLength = tlvLen;

            CHK_RET(BuildOpTopoResTlvParam(algName, serverAndsuperPodToRank, serverAndsuperPodToRankDevice, tlvLen));
            paramTask.topoInfo.serverAndsuperPodRank = reinterpret_cast<u64>(serverAndsuperPodToRankDevice.ptr());
            paramTask.topoInfo.serverAndsuperPodRankLength = tlvLen;

            CHK_RET(
                BuildOpTopoResVectorTlvParam(algName, CommPlaneSubGroupVector, complanSubGroupRankDeviceMem, tlvLen));
            paramTask.hierarchicalAlgInfo.commplaneSubGroupRank
                = reinterpret_cast<u64>(complanSubGroupRankDeviceMem.ptr());
            paramTask.hierarchicalAlgInfo.commplaneSubGroupRankLength = tlvLen;

            CHK_RET(BuildHierarchicalAlgOption(algName, hierarchicalAlgOptionVecDeviceMem, tlvLen));
            paramTask.hierarchicalAlgInfo.hierarchicalAlgOptionVec
                = reinterpret_cast<u64>(hierarchicalAlgOptionVecDeviceMem.ptr());
            paramTask.hierarchicalAlgInfo.hierarchicalAlgOptionNum = tlvLen;

            u64 len = isUsedRdmaPairVec.size() * sizeof(u32); // key-value，都为u32
            CopyVectorToDeviceMem(len, IsUsedRdmaPairVecMem, isUsedRdmaPairVec);
            paramTask.topoInfo.isUsedRdmaRankPair = reinterpret_cast<u64>(IsUsedRdmaPairVecMem.ptr());
            paramTask.topoInfo.isUsedRdmaRankPairNum = isUsedRdmaPairVec.size();

            len = pairLinkCounterVec.size() * sizeof(u32); // key-value，都为u32
            CopyVectorToDeviceMem(len, pairLinkCounterVecMem, pairLinkCounterVec);
            paramTask.topoInfo.pairLinkCounter = reinterpret_cast<u64>(pairLinkCounterVecMem.ptr());
            paramTask.topoInfo.pairLinkCounterNum = pairLinkCounterVec.size();
        }
    }

    static char lockBuffer[128];
    paramTask.lockAddr = reinterpret_cast<u64>(lockBuffer);

    // 适配HcclOpResParam零拷贝数据结构变更
    for (int i = 0; i < AICPU_ZERO_COPY_MAX_DEVICE_NUM_A3; i++) {
        paramTask.zeroCopyIpcPtrs[i] = 0x0;
    }
    return HCCL_SUCCESS;
}

uint8_t g_msgArea[COMM_MAX_WORK_SPACE_SIZE];
uint8_t g_msgAreaBak[COMM_MAX_WORK_SPACE_SIZE];

HcclResult GetOpExecCtrlCmdStub1(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kExit;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub2(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kStopExec;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub3(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kNone;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub4(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kRetry;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub5(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kStopLaunch;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub6(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::NsStopLaunch;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub8(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kClear;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub9(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::NsStopLaunch;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub10(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kChangeLink;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub11(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::kReportRetryErr;
    return HCCL_SUCCESS;
}

HcclResult GetOpExecCtrlCmdStub12(AicpuHdc* aicpuHdc, std::shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    (void)h2dTransfer;
    cmd = KfcCommand::NsStopLaunch;
    return HCCL_SUCCESS;
}

struct TestTilingData {
    uint32_t version;
    uint32_t commCnt;
    Mc2ServerCfg serverCfg;
    Mc2HcommCfg cfg1;
    Mc2HcommCfg cfg2;
};

DevType GetDevTypeStub() { return DevType::DEV_TYPE_910_93; }

struct ArgsInput {
    uint64_t inputDesc;
    void* context1;
    void* context2;
    void* workspace;
    void* tilingData;
};

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_Mc2api_Test_1)
{
    HcclOpResParam context1;
    HcclOpResParam context2;
    memset(&context1, 0, sizeof(HcclOpResParam));
    memset(&context2, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_unfold5";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context2.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    std::vector<void*> bufferVec2;
    std::string hcomId2 = "hcom_aicpu_unfold6";
    TestConstructHcclOpResParam(context2, bufferVec2, hcomId2, sqeCqeContext);
    context2.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context2.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context2.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev1 = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context2.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev1.ptr());
    context2.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context1, false));
    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context2, false));
    u64* a = (u64*)malloc(1024 * 1024);
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    memset_s(g_msgAreaBak, sizeof(g_msgAreaBak), 0, sizeof(g_msgAreaBak));

    u64 newAddr = (u64)g_msgArea;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = (HcclMsgAreaForTest*)newAddr;
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLGATHER;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 16;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    u64 newAddr2 = (u64)g_msgAreaBak;
    if (newAddr2 & 0x1ff) {
        newAddr2 = (newAddr2 & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgAreaBak = (HcclMsgAreaForTest*)newAddr2;
    // commit
    hcclMsgAreaBak->sendMsgList[0].commType = HCCL_CMD_ALLGATHER;
    hcclMsgAreaBak->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgAreaBak->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgAreaBak->sendMsgList[0].dataCnt = 16;
    hcclMsgAreaBak->sendMsgList[0].repeatCnt = 1;
    hcclMsgAreaBak->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgAreaBak->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgAreaBak->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgAreaBak->sendMsgList[0]));
    // finilze
    hcclMsgAreaBak->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgAreaBak->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgAreaBak->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgAreaBak->sendMsgList[1]));

    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);
    context2.mc2WorkSpace.workSpace = (u64)g_msgAreaBak;
    context2.mc2WorkSpace.workSpaceSize = sizeof(g_msgAreaBak);
    TestTilingData tilingData;
    memset_s(&tilingData, sizeof(TestTilingData), 0, sizeof(TestTilingData));
    tilingData.version = 2;
    tilingData.commCnt = 2;
    strcpy(tilingData.cfg1.groupName, "hcom_aicpu_unfold5\0");
    strcpy(tilingData.cfg1.algConfig, "AlltoAll=level0:fullmesh;level1:pairwise\0");
    tilingData.cfg1.opType = 8;
    strcpy(tilingData.cfg2.groupName, "hcom_aicpu_unfold6\0");
    strcpy(tilingData.cfg2.algConfig, "AllGather=level0:doublering\0");
    tilingData.cfg2.opType = 6;
    tilingData.serverCfg.debugMode = MC2_DEBUG_PRINT_BUFF;

    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);
    hccl::HcclCommAicpu* hcclCommAicpu2 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId2);
    EXPECT_NE(hcclCommAicpu2, nullptr);
    hcclCommAicpu2->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId2);
    MOCKER(AddTaskForHcclMsgV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DispatcherAiCpu::LaunchTask).stubs().will(returnValue(0));
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 2;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(desc));
    ArgsInput curArgs = {inputDesc, (void*)&context1, (void*)&context2, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));

    for (auto buff : bufferVec1) {
        free(buff);
    }
    for (auto buff : bufferVec2) {
        free(buff);
    }
    free(a);
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    free(reinterpret_cast<void*>(context2.localWindowsIn));
    free(reinterpret_cast<void*>(context2.localWindowsOut));
    free(reinterpret_cast<void*>(context2.hostStateInfo));
    free(reinterpret_cast<void*>(context2.aicpuStateInfo));
    aicpuCustomDev.free();
    aicpuCustomDev1.free();
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_Mc2api_Test_1_v1)
{
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_unfold7";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context1, false));

    u64* a = (u64*)malloc(1024 * 1024);
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    u64 newAddr = (u64)g_msgArea;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = (HcclMsgAreaForTest*)newAddr;
    HcclMsgForTest* hcclMsg = &(hcclMsgArea->sendMsgList[0]);
    hcclMsg->version = 1;
    HcclMsgV1ForTest* hcclMsg0 = reinterpret_cast<HcclMsgV1ForTest*>(hcclMsg);
    // commit
    hcclMsg0->commType = HCCL_CMD_ALLGATHER;
    hcclMsg0->hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsg0->dataCnt = 16;
    hcclMsg0->valid = HCCL_MSG_VALID_MASK;
    hcclMsg0->repeatCnt = 1;
    hcclMsg0->sendBuffer = uint64_t(a);
    hcclMsg0->recvBuffer = uint64_t(a);
    hcclMsg0->xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsg = &(hcclMsgArea->sendMsgList[1]);
    hcclMsg->version = 1;
    HcclMsgV1ForTest* hcclMsg1 = reinterpret_cast<HcclMsgV1ForTest*>(hcclMsg);
    hcclMsg1->commType = HCCL_CMD_FINALIZE;
    hcclMsg1->valid = HCCL_MSG_VALID_MASK;
    hcclMsg1->xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);

    Mc2InitTilingInner tilingData;
    std::memset(&tilingData, 0, sizeof(Mc2InitTilingInner));
    tilingData.version = 100;
    tilingData.debugMode = MC2_DEBUG_PRINT_BUFF;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;

    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910_93));
    MOCKER(AddTaskForHcclMsgV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(ParseCcOpTilingData).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DispatcherAiCpu::LaunchTask).stubs().will(returnValue(0));

    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 3;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(desc));
    ArgsInput curArgs = {inputDesc, (void*)&context1, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
    for (auto buff : bufferVec1) {
        free(buff);
    }
    free(a);
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    aicpuCustomDev.free();
}

hccl::AlgResourceResponse g_resourceForTest;
HcclResult GetAlgResponseResStub(
    HcclCommAicpu* tmpPtr, const std::string& newTag, const std::string& algName, const OpParam& opParam,
    const HcclOpResParam* commParam, std::unique_ptr<CollExecutorBase>& executor, AlgResourceResponse*& algResResponse)
{
    Stream tmp;
    g_resourceForTest.slaveStreams.push_back(tmp);
    algResResponse = &g_resourceForTest;
    return HCCL_SUCCESS;
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_Mc2api_Test_BatchWrite)
{
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_comm111";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context1, false));

    u64* a = (u64*)malloc(1024 * 1024);
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    u64 newAddr = (u64)g_msgArea;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    const u32 queueNum = 2U;
    HcclMsgAreaForMultiQueForTest* hcclMsgArea = (HcclMsgAreaForMultiQueForTest*)newAddr;
    HcclMsgV1ForTest* tmp;
    for (u32 i = 0; i < queueNum; ++i) {
        // commit
        tmp = reinterpret_cast<HcclMsgV1ForTest*>(&(hcclMsgArea->sendMsgList[i][0]));
        tmp->version = 1;
        tmp->commType = HCCL_CMD_BATCH_WRITE;
        tmp->opType = HCCL_REDUCE_SUM;
        tmp->hcclDataType = HCCL_DATA_TYPE_FP16;
        tmp->dataCnt = 16;
        tmp->valid = HCCL_MSG_VALID_MASK;
        tmp->repeatCnt = 1;
        tmp->sendBuffer = uint64_t(a);
        tmp->recvBuffer = uint64_t(a);
        tmp->xorCheck = GenXorStub((HcclMsgForTest*)tmp);
        // barrier
        tmp = reinterpret_cast<HcclMsgV1ForTest*>(&(hcclMsgArea->sendMsgList[i][1]));
        tmp->version = 1;
        tmp->commType = HCCL_CMD_BARRIER;
        tmp->valid = HCCL_MSG_VALID_MASK;
        tmp->xorCheck = GenXorStub((HcclMsgForTest*)tmp);
        // finalize
        tmp = reinterpret_cast<HcclMsgV1ForTest*>(&(hcclMsgArea->sendMsgList[i][2]));
        tmp->version = 1;
        tmp->commType = HCCL_CMD_FINALIZE;
        tmp->valid = HCCL_MSG_VALID_MASK;
        tmp->xorCheck = GenXorStub((HcclMsgForTest*)tmp);
    }

    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);

    Mc2InitTilingInner tilingData;
    std::memset(&tilingData, 0, sizeof(Mc2InitTilingInner));
    tilingData.version = 100;
    tilingData.debugMode = MC2_DEBUG_PRINT_BUFF;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.commBlockNum = 1;
    tilingData.queueNum = queueNum;

    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910_93));
    MOCKER_CPP(&HcclCommAicpu::GetAlgResponseRes).stubs().will(invoke(GetAlgResponseResStub));
    MOCKER(OrchestrateSdmaSqe).stubs().will(returnValue(0));
    MOCKER_CPP(&DispatcherAiCpu::LaunchTask).stubs().will(returnValue(0));

    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 3;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(desc));
    ArgsInput curArgs = {inputDesc, (void*)&context1, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(RunAicpuKfcSrvLaunch(args), 0);

    MOCKER(BarrierProcess).stubs().will(returnValue(HCCL_E_AGAIN));
    EXPECT_NE(RunAicpuKfcSrvLaunch(args), 0);

    for (auto buff : bufferVec1) {
        free(buff);
    }
    free(a);
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    aicpuCustomDev.free();
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_Mc2api_Test_OrchestrateSdmaSqe)
{
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_comm111";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);

    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);

    HcclComStreamInfo streamInfo;
    streamInfo.actualStreamId = 1;
    streamInfo.sqId = 1;
    streamInfo.sqDepth = 100;
    streamInfo.sqBaseAddr = &streamInfo;
    streamInfo.logicCqId = 1;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(0U, 0U, &sqeCqeCtx);
    hcclCommAicpu1->slaveStreams_.push_back(stream);
    OpParam param;
    char sqe[64];
    param.BatchWriteDataDes.queueIdx = 0U;
    param.BatchWriteDataDes.itemNum = 1U;
    param.inputPtr = sqe;
    extern void log_level_set_stub(s32 log_level);
    log_level_set_stub(0);
    EXPECT_EQ(OrchestrateSdmaSqe(param, *hcclCommAicpu1), HCCL_SUCCESS);
    log_level_set_stub(3);

    AicpuKfcRpcServerV2* rpc = GetCommRpcServer(0);
    EXPECT_NE(rpc, nullptr);
    rpc->PrintAllHcclMsgArea(1);
    for (auto buff : bufferVec1) {
        free(buff);
    }
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_Mc2api_Test_NS)
{
    KFCTaskV2 task;
    HcclOpResParam context1;
    HcclOpResParam context2;
    memset(&context1, 0, sizeof(HcclOpResParam));
    memset(&context2, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_unfold8";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context2.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    std::vector<void*> bufferVec2;
    std::string hcomId2 = "hcom_aicpu_unfold9";
    TestConstructHcclOpResParam(context2, bufferVec2, hcomId2, sqeCqeContext);
    context2.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context2.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context2.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev1 = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context2.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev1.ptr());
    context2.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context1, false));
    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context2, false));
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    memset_s(g_msgAreaBak, sizeof(g_msgAreaBak), 0, sizeof(g_msgAreaBak));
    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);
    context2.mc2WorkSpace.workSpace = (u64)g_msgAreaBak;
    context2.mc2WorkSpace.workSpaceSize = sizeof(g_msgAreaBak);
    TestTilingData tilingData;
    tilingData.version = 2;
    tilingData.commCnt = 2;
    strcpy(tilingData.cfg1.groupName, "hcom_aicpu_unfold8\0");
    strcpy(tilingData.cfg1.algConfig, "AlltoAll=level0:fullmesh;level1:pairwise\0");
    tilingData.cfg1.opType = 8;
    strcpy(tilingData.cfg2.groupName, "hcom_aicpu_unfold9\0");
    strcpy(tilingData.cfg2.algConfig, "AllGather=level0:doublering\0");
    tilingData.cfg2.opType = 6;
    tilingData.serverCfg.debugMode = 0;
    task.inputA = 0;
    task.outputC = 0;
    task.ctxNum = 2;
    task.context[0] = reinterpret_cast<u64>(&context1);
    task.context[1] = reinterpret_cast<u64>(&context2);
    task.tilingData = reinterpret_cast<u64>(&tilingData);
    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);
    hccl::HcclCommAicpu* hcclCommAicpu2 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId2);
    EXPECT_NE(hcclCommAicpu2, nullptr);
    hcclCommAicpu2->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId2);

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub9));
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2(&task), 301);
    hcclCommAicpu1->SetNsStopLaunchStatus(false);
    hcclCommAicpu2->SetNsStopLaunchStatus(false);
    for (auto buff : bufferVec1) {
        free(buff);
    }
    for (auto buff : bufferVec2) {
        free(buff);
    }
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    free(reinterpret_cast<void*>(context2.localWindowsIn));
    free(reinterpret_cast<void*>(context2.localWindowsOut));
    free(reinterpret_cast<void*>(context2.hostStateInfo));
    free(reinterpret_cast<void*>(context2.aicpuStateInfo));
    aicpuCustomDev.free();
    aicpuCustomDev1.free();
}

// 0 group 0 innersync 1 handleid 0
// 1 group 1  alltoallv
// 2 group 1 innersync 0 handleid 1
// 3 group 0 alltoallv
// 4 group 0 finialize
// 5 group 1 false
// 6 group 1 innersync 0 handleid 1
// 7 group 1 finializa

static int loopCnt = 0;
static int handleIdCnt = 0;

static bool ReadAddrMsgStub1(
    AicpuKfcRpcServerV2* tmp, HcclApi::HcclMsg* tmpMsg, HcclApi::HcclMsg* msgList, u32 queueIdx, uint32_t msgPos)
{
    HcclMsgForTest* hcclMsg = (HcclMsgForTest*)tmpMsg;
    memset(hcclMsg, 0, sizeof(HcclMsgForTest));
    hcclMsg->hcclDataType = HCCL_DATA_TYPE_INT8;
    hcclMsg->version = 0;
    if (loopCnt == 4 || loopCnt >= 7) {
        hcclMsg->commType = HCCL_CMD_FINALIZE;
    } else if (loopCnt == 0 || loopCnt == 2 || loopCnt == 6) {
        hcclMsg->commType = HCCL_CMD_INTER_GROUP_SYNC;
        hcclMsg->commDepGroupID = loopCnt == 0 ? 1 : 0;
        hcclMsg->commDepHandleID = loopCnt == 0 ? 0 : 1;
    } else if (loopCnt == 5) {
        loopCnt++;
        return false;
    } else {
        hcclMsg->selfHandleID = handleIdCnt;
        hcclMsg->commType = HCCL_CMD_ALLTOALLV;
        hcclMsg->opType = HCCL_REDUCE_RESERVED;
        hcclMsg->repeatCnt = 1;
        hcclMsg->dataCnt = 2048;
        hcclMsg->strideCount = 0;
        hcclMsg->hcclDataType = HCCL_DATA_TYPE_FP16;
        handleIdCnt++;
    }
    loopCnt++;
    return true;
}

static void
AddOneRecordSqeV1Stub(uint16_t streamId, uint16_t taskId, u64 notifyId, const uint8_t* sqeIn, uint8_t* sqeType)
{
    return;
}

static void AddOneWaitStartSqeStub(
    uint16_t streamId, uint16_t taskId, u64 waitAddr, u64 curTurnCntAddr, bool last,
    rtStarsCcoreWaitStartSqe_t* const sqe, uint8_t* sqeType)
{
    return;
}

static void AddOneWriteValueStartSqeStub(
    uint16_t streamId, uint16_t taskId, u64 writeAddr, u64 valueAddr, rtStarsCcoreWriteValueSqe_t* const sqe,
    uint8_t* sqeType)
{
    return;
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_Mc2api_2)
{
    KFCTaskV2 task;
    HcclOpResParam context1;
    HcclOpResParam context2;
    memset(&context1, 0, sizeof(HcclOpResParam));
    memset(&context2, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_unfold10";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context2.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    std::vector<void*> bufferVec2;
    std::string hcomId2 = "hcom_aicpu_unfold11";
    TestConstructHcclOpResParam(context2, bufferVec2, hcomId2, sqeCqeContext);
    context2.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context2.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context2.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev1 = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context2.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev1.ptr());
    context2.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context1, false));
    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context2, false));
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    memset_s(g_msgAreaBak, sizeof(g_msgAreaBak), 0, sizeof(g_msgAreaBak));
    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);
    context2.mc2WorkSpace.workSpace = (u64)g_msgAreaBak;
    context2.mc2WorkSpace.workSpaceSize = sizeof(g_msgAreaBak);
    TestTilingData tilingData;
    tilingData.version = 2;
    tilingData.commCnt = 2;
    strcpy(tilingData.cfg1.groupName, "hcom_aicpu_unfold10\0");
    strcpy(tilingData.cfg1.algConfig, "AlltoAll=level0:fullmesh;level1:pairwise\0");
    tilingData.cfg1.opType = 8;
    strcpy(tilingData.cfg2.groupName, "hcom_aicpu_unfold11\0");
    strcpy(tilingData.cfg2.algConfig, "AlltoAll=level0:fullmesh;level1:pairwise\0");
    tilingData.cfg2.opType = 8;
    tilingData.serverCfg.debugMode = 0;
    task.inputA = 0;
    task.outputC = 0;
    task.ctxNum = 2;
    task.context[0] = reinterpret_cast<u64>(&context1);
    task.context[1] = reinterpret_cast<u64>(&context2);
    task.tilingData = reinterpret_cast<u64>(&tilingData);
    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);
    hccl::HcclCommAicpu* hcclCommAicpu2 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId2);
    EXPECT_NE(hcclCommAicpu2, nullptr);
    hcclCommAicpu2->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId2);
    MOCKER_CPP(&AicpuKfcRpcServerV2::ReadAddrMsg).stubs().will(invoke(ReadAddrMsgStub1));
    MOCKER(AddOneRecordSqeV1).stubs().will(invoke(AddOneRecordSqeV1Stub));
    MOCKER(AddOneWaitStartSqe).stubs().will(invoke(AddOneWaitStartSqeStub));
    MOCKER_CPP(&Stream::GetNextSqeBufferAddr).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::GetAlgResponseRes).stubs().will(returnValue(0));
    MOCKER_CPP(&DispatcherAiCpu::LaunchTask).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::Orchestrate).stubs().will(returnValue(0));
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2(&task), (u32)HCCL_SUCCESS);
    for (auto buff : bufferVec1) {
        free(buff);
    }
    for (auto buff : bufferVec2) {
        free(buff);
    }
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    free(reinterpret_cast<void*>(context2.localWindowsIn));
    free(reinterpret_cast<void*>(context2.localWindowsOut));
    free(reinterpret_cast<void*>(context2.hostStateInfo));
    free(reinterpret_cast<void*>(context2.aicpuStateInfo));
    aicpuCustomDev.free();
    aicpuCustomDev1.free();
}

static int failLoopCnt = 0;

static bool ReadAddrMsgStubFalse(
    AicpuKfcRpcServerV2* tmp, HcclApi::HcclMsg* tmpMsg, HcclMsgForTest* msgList, u32 queueIdx, uint32_t msgPos)
{
    failLoopCnt++;
    if (failLoopCnt == 1) {
        HcclMsgForTest* hcclMsg = (HcclMsgForTest*)tmpMsg;
        hcclMsg->commType = HCCL_CMD_ALLGATHER;
        hcclMsg->opType = HCCL_REDUCE_RESERVED;
        hcclMsg->repeatCnt = 1;
        hcclMsg->dataCnt = 2048;
        hcclMsg->hcclDataType = HCCL_DATA_TYPE_FP16;
        hcclMsg->strideCount = 0;
        return true;
    }
    return false;
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_Mc2api_timeout)
{
    KFCTaskV2 task;
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_unfold12";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    EXPECT_EQ(0, AicpuHcclProcess::AicpuRpcResInitV2(&context1, false));
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);
    TestTilingData tilingData;
    tilingData.version = 2;
    tilingData.commCnt = 1;
    strcpy(tilingData.cfg1.groupName, "hcom_aicpu_unfold12\0");
    strcpy(tilingData.cfg1.algConfig, "AllGather=level0:doublering\0");
    tilingData.cfg1.opType = 6;
    tilingData.serverCfg.debugMode = 0;
    task.inputA = 0;
    task.outputC = 0;
    task.ctxNum = 1;
    task.context[0] = reinterpret_cast<u64>(&context1);
    task.tilingData = reinterpret_cast<u64>(&tilingData);
    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);
    MOCKER_CPP(&AicpuKfcRpcServerV2::ReadAddrMsg).stubs().will(invoke(ReadAddrMsgStubFalse));
    MOCKER(AddOneRecordSqeV1).stubs().will(invoke(AddOneRecordSqeV1Stub));
    MOCKER(AddOneWaitStartSqe).stubs().will(invoke(AddOneWaitStartSqeStub));
    MOCKER_CPP(&Stream::GetNextSqeBufferAddr).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::GetAlgResponseRes).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::Orchestrate).stubs().will(returnValue(0));
    MOCKER_CPP(&DispatcherAiCpu::LaunchTask).stubs().will(returnValue(0));
    MOCKER(&RunRpcServerInnerProcessV2).stubs().will(returnValue(9));
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2(&task), (u32)HCCL_E_TIMEOUT);
    uint16_t streamId = 1;
    uint16_t flipNum = 2;
    uint16_t taskId = 1;
    const uint8_t* sqeIn;
    uint8_t* sqeType;
    for (auto buff : bufferVec1) {
        free(buff);
    }
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    aicpuCustomDev.free();
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_aicpuServer)
{
    AicpuKfcRpcServerV2 server;
    HcclMsgForTest msg;
    memset(&msg, 0, sizeof(HcclMsgForTest));
    msg.commType = HCCL_CMD_ALLTOALLV;
    msg.repeatCnt = 1;
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    server.Init({reinterpret_cast<u64>(g_msgArea), sizeof(g_msgArea)});
    server.SetMsgPos(0, 1);
    server.GetMsgPos();
    server.GetFinishAddr(0);
    server.GetFinishAddr(HCCL_MSG_CNT);
    server.GetCommitareaAddr(0);
    server.GetCommitareaAddr(HCCL_MSG_CNT);
    CommonHcclMsg commonHcclMsg;
    GetCommonHcclMsg((HcclApi::HcclMsg*)&msg, &commonHcclMsg, 0UL);
    server.SetMsgRepeatCnt(commonHcclMsg.repeatCnt);
    server.GetMsgRepeatCnt(server.GetMsgPos());
    server.ReadAddrMsg((HcclApi::HcclMsg*)&msg, (HcclApi::HcclMsg*)&msg, 0, 0, 2);
    MOCKER_CPP(&Stream::GetNextSqeBufferAddr).stubs().will(returnValue(0));
    MOCKER(AddOneWriteValueStartSqe).stubs().will(invoke(AddOneWriteValueStartSqeStub));
    Stream stream;
    HcclMsgAreaForTest* tmp = (HcclMsgAreaForTest*)g_msgArea;
    server.AddCcoreNotify(0, reinterpret_cast<u64>(&(tmp->finishedTurnCnt[0])), 0, &stream);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_attachedStream)
{
    hccl::HcclCommAicpu hcclCommAicpu;
    AlgResourceResponse algResource;
    MOCKER(GetWorkflowMode)
        .stubs()
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE))
        .then(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));

    hcclCommAicpu.LaunchSlaveStreamTask(algResource);
    hcclCommAicpu.LaunchSlaveStreamTask(algResource);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_computeDataSize)
{
    u32 rankSize = 2;
    u64 count = 1;
    u32 dataSize = 1; // U8 是一字节
    HcclDataType type = HCCL_DATA_TYPE_INT8;

    MOCKER(&DataUnitSize).stubs().will(returnValue(0)).then(returnValue(1));

    u64 inputSize = 0;
    u64 outputSize = 0;
    EXPECT_EQ(
        AicpuHcclProcess::CalcDataSize(HcclCMDType::HCCL_CMD_ALLGATHER, type, count, rankSize, inputSize, outputSize),
        HCCL_E_PARA);

    EXPECT_EQ(
        AicpuHcclProcess::CalcDataSize(HcclCMDType::HCCL_CMD_ALLGATHER, type, count, rankSize, inputSize, outputSize),
        HCCL_SUCCESS);
    EXPECT_EQ(inputSize, count * dataSize);
    EXPECT_EQ(outputSize, rankSize * count * dataSize);

    EXPECT_EQ(
        AicpuHcclProcess::CalcDataSize(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, type, count, rankSize, inputSize, outputSize),
        HCCL_SUCCESS);
    EXPECT_EQ(inputSize, rankSize * count * dataSize);
    EXPECT_EQ(outputSize, count * dataSize);

    EXPECT_EQ(
        AicpuHcclProcess::CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, type, count, rankSize, inputSize, outputSize),
        HCCL_SUCCESS);
    EXPECT_EQ(inputSize, count * dataSize);
    EXPECT_EQ(outputSize, count * dataSize);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_aicpuServer1)
{
    AicpuKfcRpcServerV2 server;
    HcclMsgForTest msg;
    memset(&msg, 0, sizeof(HcclMsgForTest));
    server.Init({reinterpret_cast<u64>(g_msgArea), sizeof(g_msgArea)});
    server.SetMsgPos(0, 1);
    server.GetMsgPos();
    server.GetFinishAddr(0);
    server.GetFinishAddr(HCCL_MSG_CNT);
    server.GetCommitareaAddr(0);
    server.GetCommitareaAddr(HCCL_MSG_CNT);
    CommonHcclMsg commonHcclMsg;
    GetCommonHcclMsg((HcclApi::HcclMsg*)&msg, &commonHcclMsg, 0UL);
    server.SetMsgRepeatCnt(commonHcclMsg.repeatCnt);
    server.GetMsgRepeatCnt(server.GetMsgPos());
    HcclApi::HcclMsg(*msgLists)[HCCL_MSG_CNT] = server.GetMsgWorkSpace();
    ((HcclMsgForTest*)(msgLists[0]))->commType = HCCL_CMD_ALLTOALLV;
    ((HcclMsgForTest*)(msgLists[0]))->repeatCnt = 1;
    ((HcclMsgForTest*)(msgLists[0]))->valid = HCCL_MSG_VALID_MASK;
    server.ReadAddrMsg((HcclApi::HcclMsg*)&msg, (HcclApi::HcclMsg*)&msg, 0, 0, 2);
    MOCKER_CPP(&Stream::GetNextSqeBufferAddr).stubs().will(returnValue(0));
    MOCKER(AddOneWriteValueStartSqe).stubs().will(invoke(AddOneWriteValueStartSqeStub));
    Stream stream;
    HcclMsgAreaForTest* tmp = (HcclMsgAreaForTest*)g_msgArea;
    server.AddCcoreNotify(0, reinterpret_cast<u64>(&(tmp->finishedTurnCnt[0])), 0, &stream);
}

// 覆盖率用例
TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_aicpuServer2)
{
    AicpuKfcRpcServerV2 server;
    HcclMsgForTest msg;
    memset(&msg, 0, sizeof(HcclMsgForTest));
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    server.Init({reinterpret_cast<u64>(g_msgArea), sizeof(g_msgArea)});
    server.SetMsgPos(0, 1);
    server.GetMsgPos();
    server.GetFinishAddr(0);
    server.GetFinishAddr(HCCL_MSG_CNT);
    server.GetCommitareaAddr(0);
    server.GetCommitareaAddr(HCCL_MSG_CNT);
    CommonHcclMsg commonHcclMsg;
    GetCommonHcclMsg((HcclApi::HcclMsg*)&msg, &commonHcclMsg, 0UL);
    server.SetMsgRepeatCnt(commonHcclMsg.repeatCnt);
    server.GetMsgRepeatCnt(server.GetMsgPos());
    HcclApi::HcclMsg(*msgLists)[HCCL_MSG_CNT] = server.GetMsgWorkSpace();
    ((HcclMsgForTest*)(msgLists[0]))->commType = HCCL_CMD_ALLTOALLV;
    ((HcclMsgForTest*)(msgLists[0]))->hcclDataType = HCCL_DATA_TYPE_INT8;
    ((HcclMsgForTest*)(msgLists[0]))->repeatCnt = 1;
    ((HcclMsgForTest*)(msgLists[0]))->valid = HCCL_MSG_VALID_MASK;
    ((HcclMsgForTest*)(msgLists[0]))->xorCheck = GenXorStub((HcclMsgForTest*)(msgLists[0]));
    server.ReadAddrMsg((HcclApi::HcclMsg*)&msg, (HcclApi::HcclMsg*)msgLists[0], 0, 0, 2);
    MOCKER_CPP(&Stream::GetNextSqeBufferAddr).stubs().will(returnValue(0));
    MOCKER(AddOneWriteValueStartSqe).stubs().will(invoke(AddOneWriteValueStartSqeStub));
    Stream stream;
    HcclMsgAreaForTest* tmp = (HcclMsgAreaForTest*)g_msgArea;
    server.AddCcoreNotify(0, reinterpret_cast<u64>(&(tmp->finishedTurnCnt[0])), 0, &stream);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_aicpuServer3)
{
    const uint32_t AC_MSG_VALID_MASK_TAG = 0x5CDF123A;
    AicpuKfcRpcServerV2 server;
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    server.Init({reinterpret_cast<u64>(g_msgArea), sizeof(g_msgArea)});
    HcclApi::HcclMsg(*msgLists)[HCCL_MSG_CNT] = server.GetMsgWorkSpace();
    HcclMsgAreaForTest* area = (HcclMsgAreaForTest*)msgLists[0];
    area->paramExtMsgList[0].valid = ~(u64)AC_MSG_VALID_MASK_TAG;
    server.ReadValidMsgExtArea(0, 2);
    area->paramExtMsgList[0].valid = (u64)AC_MSG_VALID_MASK_TAG;
    area->paramExtMsgList[0].sendCounts[0] = 1;
    area->paramExtMsgList[0].recvCounts[0] = 1;
    area->paramExtMsgList[0].sendOffset[0] = 1;
    area->paramExtMsgList[0].recvOffset[0] = 1;
    area->paramExtMsgList[0].xorCheck = 1;
    server.ReadValidMsgExtArea(0, 1);
    area->paramExtMsgList[0].xorCheck = (u64)AC_MSG_VALID_MASK_TAG;
    server.ReadValidMsgExtArea(0, 1);
}

// 覆盖率用例
TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_aicpuServer4)
{
    AicpuKfcRpcServerV2 server;
    server.ProcessExpectPrepareMsg(0U, 0);
    server.ProcessExpectPrepareMsg(1U, 1);
    server.ProcessExpectPrepareMsg(1U, 4);
    server.ProcessExpectPrepareMsg(2U, 1);
    server.SetNeedRetryFlag(false);
}

// 覆盖率用例
TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_aicpuServer5)
{
    AicpuKfcRpcServerV2 server;
    server.Init({reinterpret_cast<u64>(g_msgArea), sizeof(g_msgArea)});
    uint32_t TEST_AC_MSG_VALID_MASK_TAG = 0x5CDF123A;
    HcclMsgForTest rMsg;
    HcclMsgForTest msg;
    memset(&msg, 0, sizeof(HcclMsgForTest));
    memset(&rMsg, 0, sizeof(HcclMsgForTest));
    msg.valid = TEST_AC_MSG_VALID_MASK_TAG;
    msg.xorCheck = 0;
    bool needReProcess = false;

    EXPECT_EQ(server.ReadValidMsg((HcclApi::HcclMsg*)&rMsg, (HcclApi::HcclMsg*)&msg, needReProcess, 0, 2), false);
    msg.version = 0;
    msg.commType = HCCL_CMD_ALLREDUCE;
    msg.opType = HCCL_REDUCE_SUM;
    msg.hcclDataType = HCCL_DATA_TYPE_INT32;
    msg.xorCheck = GenXorStub(&msg);
    EXPECT_EQ(server.ReadAddrMsg((HcclApi::HcclMsg*)&rMsg, (HcclApi::HcclMsg*)&msg, 0, 0, 2), true);

    msg.valid = TEST_AC_MSG_VALID_MASK_TAG;
    msg.commType = HCCL_CMD_ALLTOALL;
    EXPECT_EQ(server.ReadAddrMsg((HcclApi::HcclMsg*)&rMsg, (HcclApi::HcclMsg*)&msg, 0, 0, 384), false);
}

static int loopCntV2 = 0;
static int handleIdCntV2 = 0;
static bool ReadAddrMsgStubV2(
    AicpuKfcRpcServerV2* tmp, HcclApi::HcclMsg* tmpMsg, HcclApi::HcclMsg* msgList, u32 queueIdx, uint32_t msgPos)
{
    HcclMsgForTest* hcclMsg = (HcclMsgForTest*)tmpMsg;
    if (loopCntV2 >= 1) {
        hcclMsg->commType = HCCL_CMD_FINALIZE;
        hcclMsg->version = 1;
    } else {
        HcclMsgV1ForTest* hcclMsgV1 = reinterpret_cast<HcclMsgV1ForTest*>(hcclMsg);
        hcclMsgV1->commType = HCCL_CMD_ALLTOALLV;
        hcclMsgV1->opType = HCCL_REDUCE_RESERVED;
        hcclMsgV1->dataCnt = 2048;
        hcclMsgV1->strideCount = 0;
        hcclMsgV1->hcclDataType = HCCL_DATA_TYPE_FP16;
        hcclMsgV1->repeatCnt = 1;
        hcclMsgV1->selfHandleID = handleIdCntV2;
        hcclMsgV1->seqNum = 0;
        hcclMsgV1->version = 1;
        handleIdCntV2++;
    }
    loopCntV2++;
    return true;
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2V2_1)
{
    KFCTaskV2 task;
    HcclOpResParam context1;
    HcclOpResParam context2;
    memset(&context1, 0, sizeof(HcclOpResParam));
    memset(&context2, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_comm1";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context2.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    std::vector<void*> bufferVec2;
    std::string hcomId2 = "hcom_aicpu_comm2";
    TestConstructHcclOpResParam(context2, bufferVec2, hcomId2, sqeCqeContext);
    context2.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context2.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context2.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context2.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev1 = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context2.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev1.ptr());
    context2.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    AicpuHcclProcess::AicpuRpcResInitV2(&context1, false);
    AicpuHcclProcess::AicpuRpcResInitV2(&context2, false);
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    memset_s(g_msgAreaBak, sizeof(g_msgAreaBak), 0, sizeof(g_msgAreaBak));
    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);
    context2.mc2WorkSpace.workSpace = (u64)g_msgAreaBak;
    context2.mc2WorkSpace.workSpaceSize = sizeof(g_msgAreaBak);
    task.inputA = 0;
    task.outputC = 0;
    task.ctxNum = 2;
    task.context[0] = reinterpret_cast<u64>(&context1);
    task.context[1] = reinterpret_cast<u64>(&context2);
    // task.tilingData = reinterpret_cast<u64>(&tilingData);
    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);
    hccl::HcclCommAicpu* hcclCommAicpu2 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId2);
    EXPECT_NE(hcclCommAicpu2, nullptr);
    hcclCommAicpu2->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId2);
    MOCKER_CPP(&AicpuKfcRpcServerV2::ReadAddrMsg).stubs().will(invoke(ReadAddrMsgStubV2));
    MOCKER(ParseCcOpTilingData).stubs().will(returnValue(0));
    MOCKER(AddTaskForHcclMsgV2).stubs().will(returnValue(0));
    MOCKER(AddOneRecordSqeV1).stubs().will(invoke(AddOneRecordSqeV1Stub));
    MOCKER(AddOneWaitStartSqe).stubs().will(invoke(AddOneWaitStartSqeStub));
    MOCKER_CPP(&Stream::GetNextSqeBufferAddr).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::GetAlgResponseRes).stubs().will(returnValue(0));
    MOCKER_CPP(&DispatcherAiCpu::LaunchTask).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::Orchestrate).stubs().will(returnValue(0));
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2V2(&task, nullptr), (u32)HCCL_SUCCESS);
    for (auto buff : bufferVec1) {
        free(buff);
    }
    for (auto buff : bufferVec2) {
        free(buff);
    }
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    free(reinterpret_cast<void*>(context2.localWindowsIn));
    free(reinterpret_cast<void*>(context2.localWindowsOut));
    free(reinterpret_cast<void*>(context2.hostStateInfo));
    free(reinterpret_cast<void*>(context2.aicpuStateInfo));
    aicpuCustomDev.free();
    aicpuCustomDev1.free();
}

TEST_F(AicpuUnfold_UT, RpcServerV2_PrintMsg_1)
{
    HcclMsgForTest msg;
    memset(&msg, 0, sizeof(HcclMsgForTest));
    msg.commType = HCCL_CMD_ALLREDUCE;
    msg.opType = HCCL_REDUCE_SUM;
    msg.hcclDataType = HCCL_DATA_TYPE_INT8;
    msg.version = 0;
    Mc2CcTilingInner tiling;
    tiling.skipLocalRankCopy = 0;
    tiling.skipBufferWindowCopy = 0;
    tiling.stepSize = 0;
    tiling.version = 0;
    HCCL_ERROR("tiling stepSize %u", tiling.stepSize);
    strcpy(tiling.groupName, "hcom_aicpu_comm1\0");
    strcpy(tiling.algConfig, "AllGather=level0:doublering\0");
    HcclMsgV1ForTest msgV1;
    msgV1.commType = HCCL_CMD_ALLREDUCE;
    msgV1.opType = HCCL_REDUCE_SUM;
    msgV1.hcclDataType = HCCL_DATA_TYPE_INT8;
    msgV1.ccOpTilingData = (u64)(&tiling);
    msgV1.version = 1;
    std::string str = "test";
    AicpuKfcUtils::PrintMsg(str, *((HcclApi::HcclMsg*)&msg));
    AicpuKfcUtils::PrintMsg(str, *((HcclApi::HcclMsg*)&msgV1));

    CommonHcclMsg commonHcclMsg;
    GetCommonHcclMsg((HcclApi::HcclMsg*)&msgV1, &commonHcclMsg, 0UL);
    MOCKER(GetComGroupIdx).stubs().will(returnValue(0));
    EXPECT_EQ(0, ParseCcOpTilingData(&commonHcclMsg, 0));
}

// 0 group 0 allgather
// 0 group 0 finialize
Mc2CcTilingInner strideTiling;
static int32_t StrideLoopCnt = 0;
static bool MC2APIStrideMsg(
    AicpuKfcRpcServerV2* tmp, HcclApi::HcclMsg* tmpMsg, HcclApi::HcclMsg* msgList, u32 queueIdx, uint32_t msgPos)
{
    HcclMsgForTest* hcclMsg = (HcclMsgForTest*)tmpMsg;
    switch (StrideLoopCnt) {
        case 0: {
            HcclMsgV1ForTest* hcclMsgV1 = reinterpret_cast<HcclMsgV1ForTest*>(hcclMsg);
            hcclMsgV1->commType = HCCL_CMD_ALLGATHER;
            hcclMsgV1->opType = HCCL_REDUCE_RESERVED;
            hcclMsgV1->dataCnt = 256;
            hcclMsgV1->strideCount = 1024;
            strideTiling.skipLocalRankCopy = 0;
            strideTiling.skipBufferWindowCopy = 0;
            strideTiling.stepSize = 0;
            strideTiling.version = 0;
            strcpy(strideTiling.groupName, "hcom_aicpu_comm1\0");
            strcpy(strideTiling.algConfig, "AllGather=level0:doublering\0");
            strideTiling.opType = 6;
            strideTiling.reduceType = 0;
            hcclMsgV1->ccOpTilingData = reinterpret_cast<u64>(&strideTiling);
            hcclMsgV1->hcclDataType = HCCL_DATA_TYPE_FP16;
            hcclMsgV1->repeatCnt = 1;
            hcclMsgV1->selfHandleID = 0;
            hcclMsgV1->seqNum = 0;
            hcclMsgV1->version = 1;
            break;
        }
        default: {
            hcclMsg->commType = HCCL_CMD_FINALIZE;
            hcclMsg->hcclDataType = HCCL_DATA_TYPE_FP16;
            hcclMsg->version = 0;
        }
    }
    StrideLoopCnt++;
    return true;
}

// 两个通信域各发一个cfg
TEST_F(AicpuUnfold_UT, Mc2api_StrideCount)
{
    KFCTaskV2 task;
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    std::vector<void*> bufferVec1;
    std::string hcomId1 = "hcom_aicpu_comm1";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context1, bufferVec1, hcomId1, sqeCqeContext);
    context1.winSize = 4096;
    context1.localWindowsIn = reinterpret_cast<u64>(malloc(4096));
    context1.localWindowsOut = reinterpret_cast<u64>(malloc(4096));
    context1.hostStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.aicpuStateInfo = reinterpret_cast<u64>(malloc(4096));
    context1.topoInfo.deviceType = 4;
    context1.topoInfo.topoType = 6;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context1.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context1.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    AicpuHcclProcess::AicpuRpcResInitV2(&context1, false);
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    context1.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context1.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);
    task.inputA = 0;
    task.outputC = 0;
    task.ctxNum = 1;
    task.context[0] = reinterpret_cast<u64>(&context1);
    task.tilingData = 0;

    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);

    MOCKER_CPP(&AicpuKfcRpcServerV2::ReadAddrMsg).stubs().will(invoke(MC2APIStrideMsg));
    MOCKER(AddOneRecordSqeV1).stubs().will(invoke(AddOneRecordSqeV1Stub));
    MOCKER(AddOneWaitStartSqe).stubs().will(invoke(AddOneWaitStartSqeStub));
    MOCKER_CPP(&Stream::GetNextSqeBufferAddr).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::GetAlgResponseRes).stubs().will(returnValue(0));
    MOCKER_CPP(&DispatcherAiCpu::LaunchTask).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommAicpu::Orchestrate).stubs().will(returnValue(0));
    MOCKER_CPP(&AicpuKfcRpcServerV2::ProcessExpectPrepareMsg).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2V2(&task, nullptr), (u32)HCCL_SUCCESS);
    for (auto buff : bufferVec1) {
        free(buff);
    }
    free(reinterpret_cast<void*>(context1.localWindowsIn));
    free(reinterpret_cast<void*>(context1.localWindowsOut));
    free(reinterpret_cast<void*>(context1.hostStateInfo));
    free(reinterpret_cast<void*>(context1.aicpuStateInfo));
    aicpuCustomDev.free();
}

TEST_F(AicpuUnfold_UT, MC2OpExecFsmStoppingProcess)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;

    HcclOpExecFSM state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    KfcError errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));
    EXPECT_EQ(MC2OpExecFsmStoppingProcess(*hcclCommAicpu, state, errorCode), HCCL_E_PARA);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    EXPECT_EQ(errorCode, KfcError::kExec);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub1));
    EXPECT_EQ(MC2OpExecFsmStoppingProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    EXPECT_EQ(errorCode, KfcError::kExit);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub2));
    EXPECT_EQ(MC2OpExecFsmStoppingProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED);
    EXPECT_EQ(errorCode, KfcError::kSdma);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub3));
    EXPECT_EQ(MC2OpExecFsmStoppingProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING);
    EXPECT_EQ(errorCode, KfcError::kSdma);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub4));
    EXPECT_EQ(MC2OpExecFsmStoppingProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING);
    EXPECT_EQ(errorCode, KfcError::kSdma);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub5));
    EXPECT_EQ(MC2OpExecFsmStoppingProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING);
    EXPECT_EQ(errorCode, KfcError::kSdma);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub8));
    EXPECT_EQ(MC2OpExecFsmStoppingProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    EXPECT_EQ(errorCode, KfcError::kExec);
    GlobalMockObject::verify();

    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, MC2OpExecFsmStoppedProcess)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;
    uint8_t restartCnt = 0;

    HcclOpExecFSM state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
    KfcError errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));
    EXPECT_EQ(MC2OpExecFsmStoppedProcess(*hcclCommAicpu, state, errorCode), HCCL_E_PARA);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    EXPECT_EQ(errorCode, KfcError::kExec);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub1));
    EXPECT_EQ(MC2OpExecFsmStoppedProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    EXPECT_EQ(errorCode, KfcError::kExit);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
    errorCode = KfcError::kSdma;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub2));
    EXPECT_EQ(MC2OpExecFsmStoppedProcess(*hcclCommAicpu, state, errorCode), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY);
    EXPECT_EQ(errorCode, KfcError::kNone);
    GlobalMockObject::verify();

    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, MC2OpExecFsmWaitRetryProcess)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;
    uint8_t restartCnt = 0;
    bool linkChanged = false;

    HcclOpExecFSM state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    KfcError errorCode = KfcError::kNone;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));
    EXPECT_EQ(MC2OpExecFsmWaitRetryProcess(*hcclCommAicpu, state, errorCode, linkChanged), HCCL_E_PARA);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    EXPECT_EQ(errorCode, KfcError::kExec);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    errorCode = KfcError::kNone;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub1));
    EXPECT_EQ(MC2OpExecFsmWaitRetryProcess(*hcclCommAicpu, state, errorCode, linkChanged), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    EXPECT_EQ(errorCode, KfcError::kExit);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    errorCode = KfcError::kNone;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub4));
    MOCKER_CPP(&HcclCommAicpu::CleanStream).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommAicpu::ClearStreamCqeException).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(MC2OpExecFsmWaitRetryProcess(*hcclCommAicpu, state, errorCode, linkChanged), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY);
    EXPECT_EQ(errorCode, KfcError::kNone);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    errorCode = KfcError::kNone;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub10));
    EXPECT_EQ(MC2OpExecFsmWaitRetryProcess(*hcclCommAicpu, state, errorCode, linkChanged), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_CHANGE_LINK);
    EXPECT_EQ(errorCode, KfcError::kNone);
    GlobalMockObject::verify();

    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    errorCode = KfcError::kNone;
    linkChanged = true;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub10));
    EXPECT_EQ(MC2OpExecFsmWaitRetryProcess(*hcclCommAicpu, state, errorCode, linkChanged), HCCL_SUCCESS);
    EXPECT_EQ(state, HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY);
    EXPECT_EQ(errorCode, KfcError::kNone);
    GlobalMockObject::verify();

    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, CheckRestartError)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;

    MOCKER_CPP(&HcclCommAicpu::GetOpRetryEnable).stubs().with(mockcpp::any()).will(returnValue(false));
    EXPECT_EQ(HCCL_SUCCESS, CheckRestartError(hcclCommAicpu));
    GlobalMockObject::verify();

    MOCKER_CPP(&HcclCommAicpu::GetOpRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclCommAicpu::IsTaskExceptionForHccs)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(true)); // mock hccs exception sdm
    EXPECT_EQ(HCCL_E_SUSPENDING, CheckRestartError(hcclCommAicpu));
    GlobalMockObject::verify();

    MOCKER_CPP(&HcclCommAicpu::GetOpRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclCommAicpu::IsTaskExceptionForHccs).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub3));
    EXPECT_EQ(HCCL_SUCCESS, CheckRestartError(hcclCommAicpu));
    GlobalMockObject::verify();

    MOCKER_CPP(&HcclCommAicpu::GetOpRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclCommAicpu::IsTaskExceptionForHccs).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub5));
    EXPECT_EQ(HCCL_E_SUSPENDING, CheckRestartError(hcclCommAicpu));
    GlobalMockObject::verify();

    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, Mc2RetryProcess)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;

    MOCKER_CPP(
        &HcclCommAicpu::UpdateOpExecStatus,
        HcclResult(HcclCommAicpu::*)(
            HcclOpExecFSM & fsmState, KfcStatus state, KfcError & errorCode, uint32_t retryCnt))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    RestartParam restartParam;

    restartParam.fsmState[0] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
    restartParam.errorCode[0] = KfcError::kSdma;
    EXPECT_EQ(AicpuKfcRetryProcess::RetryProcess(*hcclCommAicpu, restartParam, 0), HCCL_SUCCESS);
    EXPECT_EQ(restartParam.fsmState[0], HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING);
    EXPECT_EQ(restartParam.errorCode[0], KfcError::kSdma);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub2));
    EXPECT_EQ(AicpuKfcRetryProcess::RetryProcess(*hcclCommAicpu, restartParam, 0), HCCL_SUCCESS);
    EXPECT_EQ(restartParam.fsmState[0], HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED);
    EXPECT_EQ(restartParam.errorCode[0], KfcError::kSdma);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub2));
    EXPECT_EQ(AicpuKfcRetryProcess::RetryProcess(*hcclCommAicpu, restartParam, 0), HCCL_SUCCESS);
    EXPECT_EQ(restartParam.fsmState[0], HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY);
    EXPECT_EQ(restartParam.errorCode[0], KfcError::kNone);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub10));
    EXPECT_EQ(AicpuKfcRetryProcess::RetryProcess(*hcclCommAicpu, restartParam, 0), HCCL_SUCCESS);
    EXPECT_EQ(restartParam.fsmState[0], HcclOpExecFSM::HCCL_OP_EXEC_FSM_CHANGE_LINK);
    EXPECT_EQ(restartParam.errorCode[0], KfcError::kNone);
    GlobalMockObject::verify();

    EXPECT_EQ(AicpuKfcRetryProcess::RetryProcess(*hcclCommAicpu, restartParam, 0), HCCL_SUCCESS);
    EXPECT_EQ(restartParam.fsmState[0], HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY);
    EXPECT_EQ(restartParam.errorCode[0], KfcError::kNone);

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(invoke(GetOpExecCtrlCmdStub4));
    MOCKER_CPP(&HcclCommAicpu::CleanStream).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommAicpu::ClearStreamCqeException).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(AicpuKfcRetryProcess::RetryProcess(*hcclCommAicpu, restartParam, 0), HCCL_SUCCESS);
    EXPECT_EQ(restartParam.fsmState[0], HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY);
    EXPECT_EQ(restartParam.errorCode[0], KfcError::kNone);
    GlobalMockObject::verify();

    restartParam.fsmState[0] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    restartParam.errorCode[0] = KfcError::kExec;
    EXPECT_EQ(AicpuKfcRetryProcess::RetryProcess(*hcclCommAicpu, restartParam, 0), HCCL_E_INTERNAL);

    delete hcclCommAicpu;
}

HcclResult Mc2RetryProcessStub(HcclCommAicpu& comm, RestartParam& restartParam, uint32_t idx)
{
    restartParam.consultationResult[idx] = true;
    return HCCL_SUCCESS;
}

TEST_F(AicpuUnfold_UT, RestartProcessConsulation)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;

    bool finalizeMask[MAX_COMM_CTX_NUM] = {false, false, false};
    RestartParam restartParam;

    bool flag;
    MOCKER_CPP(&AicpuKfcRpcServerV2::Reset).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuKfcRpcServerV2::WriteRestartFlag).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuKfcRetryProcess::RetryProcess).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(RestartProcessConsulation(restartParam, flag, finalizeMask, {0}), HCCL_E_INTERNAL);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuKfcRpcServerV2::Reset).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuKfcRpcServerV2::WriteRestartFlag).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuKfcRetryProcess::RetryProcess).stubs().with(mockcpp::any()).will(invoke(Mc2RetryProcessStub));
    MOCKER(AicpuKfcRetryProcess::RetryProcess).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(RestartProcessConsulation(restartParam, flag, finalizeMask, {0}), HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuKfcRpcServerV2::Reset).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuKfcRpcServerV2::WriteRestartFlag).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuKfcRetryProcess::RetryProcess).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    restartParam.consultationResult[0] = true;
    restartParam.consultationAllEnd++;
    EXPECT_EQ(RestartProcessConsulation(restartParam, flag, finalizeMask, {0}), HCCL_SUCCESS);
    GlobalMockObject::verify();

    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, HcclOpExecFsmLaunchProcess_MC2Suspending)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;
    MOCKER_CPP(&HcclCommAicpu::OrchestrateHcclOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_SUSPENDING));
    std::string algName;
    OpParam param;
    AlgResourceResponse algResource;
    HcclOpExecFSM state;
    KfcError errorCode = KfcError::kNone;
    uint32_t beginSqePos = INVALID_UINT;
    uint32_t endSqePos = INVALID_UINT;
    std::unique_ptr<CollExecutorBase> executor;
    u32 retryCnt = 0;

    hcclCommAicpu->isDeviceMode_ = true;
    hcclCommAicpu->retryEnable_ = true;

    ASSERT_EQ(
        hcclCommAicpu->HcclOpExecFsmLaunchProcess(
            algName, param, executor, algResource, state, errorCode, beginSqePos, endSqePos, retryCnt),
        HCCL_E_SUSPENDING);

    GlobalMockObject::verify();
    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, AicpuUnfold_PrepareOpParam_alltoall)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;

    hccl::OpParam opParam;
    CommonHcclMsg hcclMsg;
    hcclMsg.commType = HcclCMDType::HCCL_CMD_ALLTOALL;
    hcclMsg.opType = HCCL_REDUCE_RESERVED;
    hcclMsg.dataCnt = 1;
    hcclMsg.hcclDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    hcclMsg.strideCount = 0;

    AicpuKfcRpcServerV2 rpc;

    PrepareOpParam(&opParam, &hcclMsg, rpc, hcclCommAicpu);
    EXPECT_EQ(opParam.All2AllDataDes.sendCount, 1);
    EXPECT_EQ(opParam.All2AllDataDes.sendType, HcclDataType::HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(opParam.All2AllDataDes.recvType, HcclDataType::HCCL_DATA_TYPE_FP32);

    hcclMsg.strideCount = 1;
    hcclMsg.repeatCnt = 1;
    hcclCommAicpu->topoInfo_.userRankSize = 8;
    PrepareOpParam(&opParam, &hcclMsg, rpc, hcclCommAicpu);
    EXPECT_EQ(opParam.opType, HcclCMDType::HCCL_CMD_ALLTOALLV);
    EXPECT_EQ(opParam.All2AllDataDes.sendType, HcclDataType::HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(opParam.All2AllDataDes.recvType, HcclDataType::HCCL_DATA_TYPE_FP32);
    EXPECT_NE(opParam.All2AllDataDes.sendCounts, nullptr);
    EXPECT_NE(opParam.All2AllDataDes.recvCounts, nullptr);
    EXPECT_NE(opParam.All2AllDataDes.sdispls, nullptr);
    EXPECT_NE(opParam.All2AllDataDes.rdispls, nullptr);

    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, AicpuUnfold_PrepareOpParam_BatchWrite)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    hcclCommAicpu->dumpDebug_ = false;
    hcclCommAicpu->dispatcher_ = nullptr;

    hccl::OpParam opParam;
    CommonHcclMsg hcclMsg;
    rtStarsMemcpyAsyncSqe_t sqe;
    hcclMsg.commType = HcclCMDType::HCCL_CMD_BATCH_WRITE;
    hcclMsg.sendBuffer = (uint64_t)&sqe;
    hcclMsg.dataCnt = 1;
    hcclMsg.hcclDataType = HCCL_DATA_TYPE_INT8;
    hcclMsg.opType = HCCL_REDUCE_MIN;

    AicpuKfcRpcServerV2 rpc;
    PrepareOpParam(&opParam, &hcclMsg, rpc, hcclCommAicpu);
    EXPECT_EQ(opParam.BatchWriteDataDes.itemNum, 1);
    EXPECT_EQ(opParam.BatchWriteDataDes.queueIdx, 3);
    delete hcclCommAicpu;
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2V2_restart_1)
{
    KFCTaskV2 task;
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    task.ctxNum = 1;
    task.context[0] = reinterpret_cast<u64>(&context1);

    MOCKER(PrepareHcommInstance).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuHcclProcess::AicpuReleaseCommbyGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::GetOpRetryEnable).stubs().will(returnValue(true));
    MOCKER(RunRpcServerLoopProcess).stubs().will(returnValue(HCCL_E_SUSPENDING));
    MOCKER(RestartProcessConsulation).stubs().will(returnValue(HCCL_E_INTERNAL));

    hccl::HcclCommAicpu hcclCommAicpu;
    (void)InsertCommInst(InsertComIdMap(""), &hcclCommAicpu, &context1);
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2V2(&task, nullptr), (u32)HCCL_E_INTERNAL);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2V2_restart_retryEnable_false)
{
    KFCTaskV2 task;
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    task.ctxNum = 1;
    task.context[0] = reinterpret_cast<u64>(&context1);

    MOCKER(PrepareHcommInstance).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuHcclProcess::AicpuReleaseCommbyGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::GetOpRetryEnable).stubs().will(returnValue(false));
    MOCKER(RunRpcServerLoopProcess).stubs().will(returnValue(HCCL_E_SUSPENDING));

    hccl::HcclCommAicpu hcclCommAicpu;
    (void)InsertCommInst(InsertComIdMap(""), &hcclCommAicpu, &context1);
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2V2(&task, nullptr), (u32)HCCL_E_SUSPENDING);
}

TEST_F(AicpuUnfold_UT, RpcServerPreCheck_restart_error)
{
    hccl::HcclCommAicpu* comm = new hccl::HcclCommAicpu;
    comm->dumpDebug_ = false;
    comm->dispatcher_ = nullptr;
    comm->GetDfxExtendInfo()->pollStatus = PollStatus::kStopAsException;

    AicpuKfcRpcServerV2 rpc;

    bool finalizeFlag = false;

    MOCKER(CheckNsCommand).stubs().will(returnValue(false));
    MOCKER(CheckRestartError).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommAicpu::GetOpRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclCommAicpu::IsTaskExceptionForHccs).stubs().with(mockcpp::any()).will(returnValue(true));

    EXPECT_EQ(HCCL_E_SUSPENDING, RpcServerPreCheck(&rpc, comm, finalizeFlag));
    GlobalMockObject::verify();

    MOCKER_CPP(&HcclCommAicpu::IsTaskExceptionForHccs).stubs().with(mockcpp::any()).will(returnValue(false));
    EXPECT_EQ(HCCL_E_INTERNAL, RpcServerPreCheck(&rpc, comm, finalizeFlag));
    GlobalMockObject::verify();

    delete comm;
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2V1_NsRecovery)
{
    HcclOpResParam context;
    std::vector<void*> bufferVec;
    std::string hcomId1 = "hcom_aicpu_unfold10";
    SqCqeContext sqeCqeContext;
    TestConstructHcclOpResParam(context, bufferVec, hcomId1, sqeCqeContext);
    context.winSize = 4096;
    context.localWindowsIn = context.localWindowsOut = context.hostStateInfo = context.aicpuStateInfo
        = reinterpret_cast<u64>(malloc(4096));
    context.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    context.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);
    TestTilingData tilingData;
    tilingData.version = 2;
    tilingData.commCnt = 1;
    strcpy(tilingData.cfg1.groupName, "hcom_aicpu_unfold10\0");
    strcpy(tilingData.cfg1.algConfig, "AlltoAll=level0:fullmesh;level1:pairwise\0");
    tilingData.cfg1.opType = 8;
    KFCTaskV2 task{};
    task.ctxNum = 1;
    task.context[0] = reinterpret_cast<u64>(&context);
    task.tilingData = reinterpret_cast<u64>(&tilingData);
    hccl::HcclCommAicpu* hcclCommAicpu1 = AicpuHcclProcess::AicpuGetCommbyGroup(hcomId1);
    EXPECT_NE(hcclCommAicpu1, nullptr);
    hcclCommAicpu1->SetDumpDebug(false);
    AicpuHcclProcess::AicpuReleaseCommbyGroup(hcomId1);

    MOCKER(CheckNsStopLaunchStatus).stubs().will(returnValue(HCCL_E_SUSPENDING));
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2(&task), 301);

    for (auto buff : bufferVec) {
        free(buff);
    }
    free(reinterpret_cast<void*>(context.localWindowsIn));
    aicpuCustomDev.free();
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2V2_NsRecovery)
{
    HcclOpResParam context;
    std::vector<void*> bufferVec;
    SqCqeContext sqeCqeContext;
    std::string hcomId = "hcom_aicpu_comm1";
    TestConstructHcclOpResParam(context, bufferVec, hcomId, sqeCqeContext);
    context.winSize = 4096;
    context.localWindowsIn = context.localWindowsOut = context.hostStateInfo = context.aicpuStateInfo
        = reinterpret_cast<u64>(malloc(4096));
    context.topoInfo.deviceType = 4;
    DeviceMem aicpuCustomDev = DeviceMem::alloc(sizeof(AicpuCustomParam));
    context.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev.ptr());
    context.aicpuCustomParamSize = sizeof(AicpuCustomParam);

    AicpuHcclProcess::AicpuRpcResInitV2(&context, false);
    context.mc2WorkSpace.workSpace = (u64)g_msgArea;
    context.mc2WorkSpace.workSpaceSize = sizeof(g_msgArea);

    KFCTaskV2 task{};
    task.ctxNum = 1;
    task.context[0] = reinterpret_cast<u64>(&context);

    MOCKER(CheckNsStopLaunchStatus).stubs().will(returnValue(HCCL_E_SUSPENDING));
    EXPECT_EQ(AicpuKfcProcess::AicpuRunRpcServerForMC2V2(&task, nullptr), 301);

    for (auto buff : bufferVec) {
        free(buff);
    }
    free(reinterpret_cast<void*>(context.localWindowsIn));
    aicpuCustomDev.free();
}

/*正式方案上库需要适配*/
// TEST_F(AicpuUnfold_UT, TestStructSize)
// {
//     EXPECT_EQ(offsetof(HcclCombinOpParam, multiServerFlag), 8872);
//     EXPECT_EQ(offsetof(HcclCombinOpParam, ibverbsData), 8880);
//     EXPECT_EQ(offsetof(HcclOpResParam, remoteRes), 2984);
// }

HcclResult HcclGetCommHandleByCtxStub(void* ctx, void** handle)
{
    *handle = ctx;
    return HCCL_SUCCESS;
}

HcclResult HcclGetTaskStatusStub(void* handle, HcclTaskStatus* status)
{
    *status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    return HCCL_SUCCESS;
}

HcclResult HcclGetTaskStatusInvalidStub(void* handle, HcclTaskStatus* status)
{
    *status = HcclTaskStatus::HCCL_CQE_ERROR;
    return HCCL_SUCCESS;
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_DecoulpledCtx)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    memset_s(g_msgAreaBak, sizeof(g_msgAreaBak), 0, sizeof(g_msgAreaBak));

    HcclMsgForTest* syncMsg = (HcclMsgForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    // group sync
    syncMsg->commType = HCCL_CMD_INTER_GROUP_SYNC;
    syncMsg->commDepGroupID = 1;
    syncMsg->commDepHandleID = 1;
    syncMsg->valid = HCCL_MSG_VALID_MASK;
    syncMsg->xorCheck = GenXorStub(syncMsg);
    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[1]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLGATHER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 255;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->selfHandleID = 0;
    msg->seqNum = 2;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // commit
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLTOALL;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 3;
    msg->selfHandleID = 1;
    msg->seqNum = 3;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // commit
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLTOALLV;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 4;
    msg->selfHandleID = 2;
    msg->seqNum = 4;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    HcclApi::HcclMsgExt* extMsg = &(((HcclMsgAreaForTest*)g_msgArea)->paramExtMsgList[3]);
    extMsg->sendCounts[0] = extMsg->sendOffset[0] = extMsg->recvCounts[0] = extMsg->recvOffset[0] = 666;
    extMsg->sendCounts[1] = extMsg->sendOffset[1] = extMsg->recvCounts[1] = extMsg->recvOffset[1] = 999;
    extMsg->valid = HCCL_MSG_VALID_MASK;
    extMsg->xorCheck = GenXorStub(extMsg, 2);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    // commit
    msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgAreaBak)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLREDUCE;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 2;
    msg->selfHandleID = 0;
    msg->seqNum = 0;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // commit
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_REDUCE_SCATTER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 5;
    msg->selfHandleID = 1;
    msg->seqNum = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 5;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    ctx1.apiCtx.rankNum = 2;
    CommKfcContext ctx2{};
    ctx2.hcclContext = 2;
    ctx2.apiCtx.workSpace = (u64)g_msgAreaBak;
    ctx2.apiCtx.rankNum = 2;
    CommKfcContext ctx3{};
    ctx3.hcclContext = 3;
    ctx3.apiCtx.workSpace = (u64)g_msgArea;
    CommKfcContext ctx4{};
    ctx4.hcclContext = 4;
    ctx4.apiCtx.workSpace = (u64)g_msgArea;
    CommKfcContext ctx5{};
    ctx5.hcclContext = 5;
    ctx5.apiCtx.workSpace = (u64)g_msgAreaBak;
    uint64_t args[6] = {inputDesc, (uint64_t)&ctx1, (uint64_t)&ctx2, (uint64_t)&ctx3, (uint64_t)&ctx4, (uint64_t)&ctx5};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    MOCKER(HcclGetTaskStatus).stubs().with(mockcpp::any()).will(invoke(HcclGetTaskStatusStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), 0);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_SeqTest)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    memset_s(g_msgAreaBak, sizeof(g_msgAreaBak), 0, sizeof(g_msgAreaBak));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLTOALLV;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->selfHandleID = 0;
    msg->seqNum = 2;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    HcclApi::HcclMsgExt* extMsg = &(((HcclMsgAreaForTest*)g_msgArea)->paramExtMsgList[0]);
    extMsg->sendCounts[0] = extMsg->sendOffset[0] = extMsg->recvCounts[0] = extMsg->recvOffset[0] = 777;
    extMsg->sendCounts[1] = extMsg->sendOffset[1] = extMsg->recvCounts[1] = extMsg->recvOffset[1] = 888;
    extMsg->valid = HCCL_MSG_VALID_MASK;
    extMsg->xorCheck = GenXorStub(extMsg, 2);
    // commit
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLTOALL;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 2;
    msg->selfHandleID = 1;
    msg->seqNum = 3;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    // commit
    msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgAreaBak)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLREDUCE;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 3;
    msg->selfHandleID = 0;
    msg->seqNum = 0;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // commit
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_REDUCE_SCATTER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 4;
    msg->selfHandleID = 1;
    msg->seqNum = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 4;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    ctx1.apiCtx.rankNum = 2;
    CommKfcContext ctx2{};
    ctx2.hcclContext = 2;
    ctx2.apiCtx.workSpace = (u64)g_msgArea;
    ctx2.apiCtx.rankNum = 2;
    CommKfcContext ctx3{};
    ctx3.hcclContext = 3;
    ctx3.apiCtx.workSpace = (u64)g_msgAreaBak;
    CommKfcContext ctx4{};
    ctx4.hcclContext = 4;
    ctx4.apiCtx.workSpace = (u64)g_msgAreaBak;
    uint64_t args[5] = {inputDesc, (uint64_t)&ctx1, (uint64_t)&ctx2, (uint64_t)&ctx3, (uint64_t)&ctx4};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    MOCKER(HcclGetTaskStatus).stubs().with(mockcpp::any()).will(invoke(HcclGetTaskStatusStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), 0);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_RetryFail)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLREDUCE;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->seqNum = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_INTERNAL);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_GroupSyncInvalidPara)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    memset_s(g_msgAreaBak, sizeof(g_msgAreaBak), 0, sizeof(g_msgAreaBak));

    HcclMsgForTest* syncMsg = (HcclMsgForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    // group sync
    syncMsg->commType = HCCL_CMD_INTER_GROUP_SYNC;
    syncMsg->commDepGroupID = 0;
    syncMsg->commDepHandleID = 1;
    syncMsg->valid = HCCL_MSG_VALID_MASK;
    syncMsg->xorCheck = GenXorStub(syncMsg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 2;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    ctx1.apiCtx.rankNum = 2;
    CommKfcContext ctx2{};
    ctx2.hcclContext = 2;
    ctx2.apiCtx.workSpace = (u64)g_msgAreaBak;
    ctx2.apiCtx.rankNum = 2;
    uint64_t args[3] = {inputDesc, (uint64_t)&ctx1, (uint64_t)&ctx2};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_PARA);

    syncMsg->commDepGroupID = 100;
    syncMsg->valid = HCCL_MSG_VALID_MASK;
    syncMsg->xorCheck = GenXorStub(syncMsg);
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_PARA);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_InvalidRankNumForAlltoallv)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLTOALLV;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->selfHandleID = 0;
    msg->seqNum = 0;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    HcclApi::HcclMsgExt* extMsg = &(((HcclMsgAreaForTest*)g_msgArea)->paramExtMsgList[0]);
    extMsg->sendCounts[0] = extMsg->sendOffset[0] = extMsg->recvCounts[0] = extMsg->recvOffset[0] = 666;
    extMsg->sendCounts[1] = extMsg->sendOffset[1] = extMsg->recvCounts[1] = extMsg->recvOffset[1] = 999;
    extMsg->valid = HCCL_MSG_VALID_MASK;
    extMsg->xorCheck = GenXorStub(extMsg, 256);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    ctx1.apiCtx.rankNum = 2000;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    MOCKER(HcclGetTaskStatus).stubs().with(mockcpp::any()).will(invoke(HcclGetTaskStatusStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_PARA);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_InvalidCmdType)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_RECEIVE;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_PARA);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_InvalidCctiling)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLGATHER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 999;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_PARA);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_PrepareMsgTimeout)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLGATHER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = 0;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_TIMEOUT);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_FinalizeMsgTimeout)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLGATHER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = 0;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_TIMEOUT);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_FinalizeTimeout)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLGATHER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    MOCKER(HcclCheckFinishByStream).stubs().will(returnValue(HCCL_E_SUSPENDING));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_TIMEOUT);
}

TEST_F(AicpuUnfold_UT, AicpuRunRpcServerForMC2_FinalizeStatusError)
{
    memset_s(g_msgArea, sizeof(g_msgArea), 0, sizeof(g_msgArea));

    // commit
    HcclMsgV1ForTest* msg = (HcclMsgV1ForTest*)&(((HcclMsgAreaForTest*)g_msgArea)->sendMsgList[0]);
    msg->version = 3;
    msg->commType = HCCL_CMD_ALLGATHER;
    msg->hcclDataType = HCCL_DATA_TYPE_FP16;
    msg->dataCnt = 16;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->repeatCnt = 2;
    msg->sendBuffer = 0x12345678;
    msg->recvBuffer = 0x12345678;
    msg->ccOpTilingData = 1;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);
    // finalize
    ++msg;
    msg->version = 3;
    msg->commType = HCCL_CMD_FINALIZE;
    msg->valid = HCCL_MSG_VALID_MASK;
    msg->xorCheck = GenXorStub((HcclMsgForTest*)msg);

    CommKfcParamDesc desc;
    desc.version = 2;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 0;
    desc.isDyn = 0;
    uint64_t inputDesc;
    memcpy_s(&inputDesc, sizeof(inputDesc), &desc, sizeof(CommKfcParamDesc));
    CommKfcContext ctx1{};
    ctx1.hcclContext = 1;
    ctx1.apiCtx.workSpace = (u64)g_msgArea;
    uint64_t args[2] = {inputDesc, (uint64_t)&ctx1};
    MOCKER(HcclGetCommHandleByCtx).stubs().with(mockcpp::any()).will(invoke(HcclGetCommHandleByCtxStub));
    MOCKER(HcclGetTaskStatus).stubs().with(mockcpp::any()).will(invoke(HcclGetTaskStatusInvalidStub));
    EXPECT_EQ(RunAicpuKfcSrvLaunch((void**)args), HCCL_E_INTERNAL);
}

/**
 * @brief AicpuRpcClearOpRes: nullptr 输入
 * 验证：tilingData=nullptr 时返回 HCCL_E_PARA
 */
TEST_F(AicpuUnfold_UT, Ut_AicpuRpcClearOpRes_NullTilingData)
{
    u32 ret = AicpuHcclProcess::AicpuRpcClearOpRes(nullptr);
    EXPECT_EQ(ret, static_cast<u32>(HCCL_E_PARA));
}

/**
 * @brief AicpuRpcClearOpRes: magic 不匹配
 * 验证：magic 不等于 HCCL_KFC_CLEAR_OP_RES_MAGIC 时返回 HCCL_E_PARA
 */
TEST_F(AicpuUnfold_UT, Ut_AicpuRpcClearOpRes_MagicMismatch)
{
    auto tilingData = std::make_unique<HcclKfcClearOpResTilingData>();
    memset_s(tilingData.get(), sizeof(*tilingData), 0, sizeof(*tilingData));
    tilingData->magic = 0xDEADBEEF; // 错误的 magic
    tilingData->tagCount = 1;

    u32 ret = AicpuHcclProcess::AicpuRpcClearOpRes(tilingData.get());
    EXPECT_EQ(ret, static_cast<u32>(HCCL_E_PARA));
}

/**
 * @brief AicpuRpcClearOpRes: tagCount=0
 * 验证：tagCount=0 时返回 HCCL_E_PARA
 */
TEST_F(AicpuUnfold_UT, Ut_AicpuRpcClearOpRes_TagCountZero)
{
    auto tilingData = std::make_unique<HcclKfcClearOpResTilingData>();
    memset_s(tilingData.get(), sizeof(*tilingData), 0, sizeof(*tilingData));
    tilingData->magic = HCCL_KFC_CLEAR_OP_RES_MAGIC;
    tilingData->tagCount = 0;

    u32 ret = AicpuHcclProcess::AicpuRpcClearOpRes(tilingData.get());
    EXPECT_EQ(ret, static_cast<u32>(HCCL_E_PARA));
}

/**
 * @brief AicpuRpcClearOpRes: tagCount 超过 MAX_BATCH
 * 验证：tagCount > MAX_BATCH 时返回 HCCL_E_PARA
 */
TEST_F(AicpuUnfold_UT, Ut_AicpuRpcClearOpRes_TagCountExceedsMax)
{
    auto tilingData = std::make_unique<HcclKfcClearOpResTilingData>();
    memset_s(tilingData.get(), sizeof(*tilingData), 0, sizeof(*tilingData));
    tilingData->magic = HCCL_KFC_CLEAR_OP_RES_MAGIC;
    tilingData->tagCount = HCCL_KFC_CLEAR_OP_RES_MAX_BATCH + 1;

    u32 ret = AicpuHcclProcess::AicpuRpcClearOpRes(tilingData.get());
    EXPECT_EQ(ret, static_cast<u32>(HCCL_E_PARA));
}

/**
 * @brief AicpuRpcClearOpRes: group 不存在（graceful 退化路径）
 * 验证：AicpuGetCommbyGroup 返回 nullptr 时返回 HCCL_SUCCESS
 */
TEST_F(AicpuUnfold_UT, Ut_AicpuRpcClearOpRes_GroupNotFound)
{
    auto tilingData = std::make_unique<HcclKfcClearOpResTilingData>();
    memset_s(tilingData.get(), sizeof(*tilingData), 0, sizeof(*tilingData));
    tilingData->magic = HCCL_KFC_CLEAR_OP_RES_MAGIC;
    tilingData->tagCount = 1;
    memcpy_s(tilingData->group, sizeof(tilingData->group), "nonexistent_group", 18);
    memcpy_s(tilingData->tags[0], TAG_MAX_LENGTH, "test_tag", 9);

    u32 ret = AicpuHcclProcess::AicpuRpcClearOpRes(tilingData.get());
    // group 不存在时 AicpuGetCommbyGroup 返回 nullptr，函数返回 SUCCESS
    EXPECT_EQ(ret, static_cast<u32>(HCCL_SUCCESS));
}
