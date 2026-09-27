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
#include <mockcpp/mokc.h>
#include <chrono>
#include <iostream>
#include <memory>

#include "primitive.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "log.h"

#include "virtual_topo.h"
#include "virtual_topo_stub.h"

#define private public
#include "topo_match_mesh.h"
#include "ins_all_to_all_sole_executor.h"
#include "ccu_temp_all_to_all_mesh2d.h"
#include "ccu_context_all_to_all_mesh2d.h"
#include "ccu_instruction_all_to_all_mesh2d.h"
#include "ccu_connection.h"
#include "ccu_transport.h"
#include "ccu_device_manager.h"
#include "ccu_component.h"
#include "ccu_res_specs_legacy.h"
#include "ccu_rep.h"
#include "ccu_ctx.h"
#include "ccu_rep_translator.h"
#include "ccu_ctx_mgr.h"
#include "ins_exe_que.h"
#include "ccu_transport_group.h"
#include "ccu_transport.h"
#include "data_type.h"
#include "dev_buffer.h"
#include "base_mem_transport.h"

using namespace Hccl;
using namespace CcuRep;

class CcuMesh2DTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuMesh1DTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuMesh1DTest tear down" << std::endl; }

    virtual void SetUp()
    {
        JfcHandle jfcHandle = 1;
        MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
        std::cout << "A Test case in CcuMesh1DTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();

        std::cout << "A Test case in CcuMesh1DTest TearDown" << std::endl;
    }
};

void MockDoOnce();
HcclResult
GetCcuRmaBufferStub(const int32_t deviceLogicId, const uint8_t dieId, std::shared_ptr<LocalUbRmaBuffer>& ccuRmaBuffer);
HcclResult AllocChannelStub(
    const int32_t deviceLogicId, const uint8_t dieId, const ChannelPara channelPara, ChannelInfo& channelInfo);
HcclResult
AllocCkeStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos);
HcclResult
AllocXnStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos);

TEST_F(CcuMesh2DTest, CCU_A2A_Mesh_template)
{
    // 创建需求资源
    RankId myRank = 0;
    u32 rankSize = 8;
    std::vector<std::vector<RankId>> tempVTopo = {{0, 1, 2, 3}, {0, 4}};
    std::map<RankId, u32> tempVirtRankMap = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}};

    // 开始执行
    std::shared_ptr<CcuTempAlltoAllMesh2D> algoTemplate
        = std::make_shared<CcuTempAlltoAllMesh2D>(myRank, rankSize, tempVTopo, tempVirtRankMap);

    // 结果验证
    EXPECT_NE(algoTemplate, nullptr);
}

TEST_F(CcuMesh2DTest, CCU_A2A_Mesh_CalcRes)
{
    // 创建需求资源
    RankId myRank = 0;
    std::vector<std::vector<RankId>> tempVTopo = {{0, 1, 2, 3}, {0, 4}};
    std::map<RankId, u32> tempVirtRankMap = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}};
    u32 rankSize = tempVirtRankMap.size();

    std::shared_ptr<CcuTempAlltoAllMesh2D> algoTemplate
        = std::make_shared<CcuTempAlltoAllMesh2D>(myRank, rankSize, tempVTopo, tempVirtRankMap);

    AlgTempResReq tempResReq;

    // 开始执行
    ASSERT_EQ(algoTemplate->CalcRes(tempResReq), HcclResult::HCCL_SUCCESS);

    // 结果验证
    EXPECT_EQ(tempResReq.queNum, 1);
    EXPECT_EQ(tempResReq.queNotifys.size(), 0);
    std::vector<RankId> expextedLinkPeers = {1, 2, 3, 4};
    EXPECT_EQ(tempResReq.links.size(), expextedLinkPeers.size());
    for (RankId rank : expextedLinkPeers) {
        EXPECT_EQ(tempResReq.links.count(rank), 1);
    }
}

TEST_F(CcuMesh2DTest, CCU_A2A_Mesh_template_run)
{
    // 创建需求资源
    RankId myRank = 0;
    std::vector<std::vector<RankId>> tempVTopo = {{0, 1, 2, 3}, {0, 4}};
    std::map<RankId, u32> tempVirtRankMap = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}};
    u32 rankSize = tempVirtRankMap.size();

    std::shared_ptr<CcuTempAlltoAllMesh2D> algoTemplate
        = std::make_shared<CcuTempAlltoAllMesh2D>(myRank, rankSize, tempVTopo, tempVirtRankMap);

    u64 sliceSize = 1024;
    RankSliceInfo sliceInfoVec;
    CollAlgOperator collAlgOp;
    collAlgOp.opType = OpType::ALLTOALL;
    collAlgOp.dataType = DataType::FP16;
    collAlgOp.dataCount = 64;
    uint64_t dataSize = collAlgOp.dataCount * 2;
    collAlgOp.inputMem = DevBuffer::Create(0x1000000, dataSize);
    collAlgOp.outputMem = DevBuffer::Create(0x2000000, dataSize * rankSize);
    collAlgOp.scratchMem = DevBuffer::Create(0x3000000, dataSize * rankSize);

    algoTemplate->SetCollOp(collAlgOp);

    TempFuncs tempFuncs;
    tempFuncs.opMode = OpMode::OPBASE;
    tempFuncs.enableCounterNotify = false;
    tempFuncs.isForepart = true; // Usr Buff to CCL Buff required
    tempFuncs.isBottom = true;   // CCL Buff to Usr Buff required

    BuffInfo buffInfo;
    ResLinks tempLinks;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    for (uint32_t dim = 0; dim < tempVTopo.size(); dim++) {
        for (auto rankIdx : tempVTopo[dim]) {
            if (rankIdx == myRank) {
                continue;
            }
            LinkData link(portType, myRank, rankIdx, 0, 0);
            tempLinks[rankIdx].push_back(link);
        }
    }

    A2ASendRecvInfo localSendRecvInfo;
    localSendRecvInfo.sendLength.emplace_back(sliceSize);
    algoTemplate->SetA2ASendRecvInfo(localSendRecvInfo);
    algoTemplate->GetScratchBufferInfo(1048576, collAlgOp.dataType);

    std::vector<InsQuePtr> tempInsQues;
    tempInsQues.push_back(std::make_shared<InsQueue>());
    ASSERT_EQ(algoTemplate->Run(tempFuncs, sliceInfoVec, buffInfo, tempLinks, tempInsQues), HcclResult::HCCL_SUCCESS);
    for (auto insQue : tempInsQues) {
        for (auto iter = insQue->Iter(); iter.HasNext(); ++iter) {
            std::cout << iter->Describe() << std::endl;
        }
    }
}

TEST_F(CcuMesh2DTest, CCU_A2A_Mesh_sole_executor)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095TwoTimesTwo(rankTable);

    RankId myRank = 0;
    u32 rankSize = 4;
    std::unique_ptr<InsAlltoAllSoleExecutor<TopoMatchConcurrMesh, CcuTempAlltoAllMesh2D>> algoExecutor(
        new InsAlltoAllSoleExecutor<TopoMatchConcurrMesh, CcuTempAlltoAllMesh2D>);

    algoExecutor->SetMyRank(myRank);
    algoExecutor->SetRankSize(rankSize);
    algoExecutor->EnableDataAllign(false);
    algoExecutor->EnableDetour(false);
    algoExecutor->SetDevType(DevType::DEV_TYPE_950);

    CollAlgOperator collAlgOp;
    collAlgOp.opType = OpType::ALLTOALL;
    collAlgOp.dataCount = 64;
    collAlgOp.all2AllDataDes.sendType = DataType::FP32;
    collAlgOp.all2AllDataDes.recvType = DataType::FP32;
    collAlgOp.all2AllDataDes.sendCount = 64;
    collAlgOp.all2AllDataDes.recvCount = 64;
    uint64_t dataSize = collAlgOp.dataCount * 2;
    collAlgOp.inputMem = DevBuffer::Create(0x1000000, dataSize);
    collAlgOp.outputMem = DevBuffer::Create(0x2000000, dataSize * rankSize);
    collAlgOp.scratchMem = DevBuffer::Create(0x3000000, dataSize * rankSize);

    CollAlgParams collAlgParams;
    collAlgParams.opMode = OpMode::OPBASE;
    collAlgParams.maxTmpMemSize = 1048576;

    CollAlgResReq resReq;
    auto ret = algoExecutor->CalcRes(&virtTopo, resReq);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS); // check return
    EXPECT_EQ(resReq.primQueueNum, 2);        // check required sub queue num

    std::shared_ptr<InsQueue> insQue(new InsQueue);

    algoExecutor->vTopo_.clear();
    algoExecutor->virtRankMap_.clear();
    algoExecutor->virtRanks_.clear();

    ret = algoExecutor->Orchestrate(&virtTopo, collAlgOp, collAlgParams, insQue);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS); // check return
    EXPECT_EQ(insQue->SizeOfSlaves(), 0);

    for (auto iter = insQue->Iter(); iter.HasNext(); ++iter) {
        std::cout << iter->Describe() << std::endl;
    }
}

HcclResult CtxMgrAllocCkeStubAlg(
    const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    ckeInfos.clear();
    ResInfo ckeInfo(0, num);
    ckeInfos.push_back(ckeInfo);
    return HcclResult::HCCL_SUCCESS;
}
HcclResult CtxMgrAllocXnStubAlg(
    const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    xnInfos.clear();
    ResInfo xnInfo(0, num);
    xnInfos.push_back(xnInfo);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrAllocResHandleStubAlg(const int32_t deviceLogicId, const CcuResReq resReq, CcuResHandle& handle)
{
    handle = reinterpret_cast<CcuResHandle>(0x100);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CtxMgrGetResourceStubAlg(const int32_t deviceLogicId, const CcuResHandle handle, CcuResRepository& ccuResRepo)
{
    ccuResRepo.blockMs[0].resize(1);
    ccuResRepo.blockMs[0][0].startId = 0;
    ccuResRepo.blockMs[0][0].num = 512;
    ccuResRepo.blockMs[1].resize(1);
    ccuResRepo.blockMs[1][0].startId = 0;
    ccuResRepo.blockMs[1][0].num = 512;

    ccuResRepo.cke[0].resize(1);
    ccuResRepo.cke[0][0].startId = 512;
    ccuResRepo.cke[0][0].num = 100;
    ccuResRepo.cke[1].resize(1);
    ccuResRepo.cke[1][0].startId = 512;
    ccuResRepo.cke[1][0].num = 100;

    ccuResRepo.blockCke[0].resize(1);
    ccuResRepo.blockCke[0][0].startId = 0;
    ccuResRepo.blockCke[0][0].num = 64;
    ccuResRepo.blockCke[1].resize(1);
    ccuResRepo.blockCke[1][0].startId = 0;
    ccuResRepo.blockCke[1][0].num = 64;

    ccuResRepo.blockLoopEngine[0].resize(1);
    ccuResRepo.blockLoopEngine[0][0].startId = 0;
    ccuResRepo.blockLoopEngine[0][0].num = 64;
    ccuResRepo.blockLoopEngine[1].resize(1);
    ccuResRepo.blockLoopEngine[1][0].startId = 0;
    ccuResRepo.blockLoopEngine[1][0].num = 64;

    ccuResRepo.gsa[0].resize(1);
    ccuResRepo.gsa[0][0].startId = 0;
    ccuResRepo.gsa[0][0].num = 100;
    ccuResRepo.gsa[1].resize(1);
    ccuResRepo.gsa[1][0].startId = 0;
    ccuResRepo.gsa[1][0].num = 100;

    ccuResRepo.xn[0].resize(1);
    ccuResRepo.xn[0][0].startId = 1024;
    ccuResRepo.xn[0][0].num = 1536;
    ccuResRepo.xn[1].resize(1);
    ccuResRepo.xn[1][0].startId = 1024;
    ccuResRepo.xn[1][0].num = 1536;

    ccuResRepo.mission.mission[0].resize(1);
    ccuResRepo.mission.mission[0][0].startId = 0;
    ccuResRepo.mission.mission[0][0].num = 1;
    ccuResRepo.mission.mission[1].resize(1);
    ccuResRepo.mission.mission[1][0].startId = 0;
    ccuResRepo.mission.mission[1][0].num = 1;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrReleaseResHandleStubAlg(const int32_t deviceLogicId, const CcuResHandle handle)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrAllocInsStubAlg(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    insInfo = ResInfo(0, num);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrReleaseInsStubAlg(const int32_t deviceLogicId, const uint8_t dieId, ResInfo& insInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrGetMissionKeyStubAlg(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& missionKey)
{
    missionKey = 0xFF;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrGetInstructionNumStubAlg(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& instrNum)
{
    instrNum = 0xFF;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResourceMangerGetLoopChannelIdStub(
    const int32_t deviceLogicId, const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channIdx)
{
    if (dstDieId == 0) {
        channIdx = 126;
    } else if (dstDieId == 1) {
        channIdx = 127;
    }
    return HcclResult::HCCL_SUCCESS;
}

TEST_F(CcuMesh2DTest, CCU_A2A_Mesh_sole_context)
{
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::Destroy).stubs();
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(CtxMgrAllocCkeStubAlg));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(CtxMgrAllocXnStubAlg));
    MOCKER_CPP(&CcuDeviceManager::GetCcuResourceSpaceTokenInfoForLocal)
        .stubs()
        .will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuDeviceManager::GetCcuResourceSpaceTokenInfo)
        .stubs()
        .will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocResHandle).stubs().will(invoke(CtxMgrAllocResHandleStubAlg));
    MOCKER(CcuDeviceManager::GetResource).stubs().will(invoke(CtxMgrGetResourceStubAlg));
    MOCKER(CcuDeviceManager::ReleaseResHandle).stubs().will(invoke(CtxMgrReleaseResHandleStubAlg));
    MOCKER(CcuDeviceManager::AllocIns).stubs().will(invoke(CtxMgrAllocInsStubAlg));
    MOCKER(CcuDeviceManager::ReleaseIns).stubs().will(invoke(CtxMgrReleaseInsStubAlg));
    MOCKER(CcuDeviceManager::GetInstructionNum).stubs().will(invoke(CtxMgrGetInstructionNumStubAlg));
    MOCKER(CcuDeviceManager::GetMissionKey).stubs().will(invoke(CtxMgrGetMissionKeyStubAlg));

    MOCKER(HrtMemcpy).stubs();

    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(AllocCkeStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(AllocXnStub));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(CcuDeviceManager::GetLoopChannelId).stubs().will(invoke(CcuResourceMangerGetLoopChannelIdStub));
    MOCKER(&CcuDeviceManager::GetXnBaseAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(&CcuDeviceManager::GetCcuResourceSpaceTokenInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MockDoOnce();

    CollAlgOperator collAlgOp;
    collAlgOp.opType = OpType::ALLTOALL;
    collAlgOp.dataType = DataType::INT8;
    collAlgOp.dataCount = 4;

    uint32_t rankId = 2;
    uint32_t rankSize = 4;
    std::vector<uint32_t> dimSize = {2, 2};
    std::vector<uint32_t> dimIds = {rankId % dimSize[0], rankId / dimSize[0]};
    std::vector<std::vector<RankId>> tempVTopo = {{0, 1}, {0, 2}};

    for (uint32_t rankId = 0; rankId < 1; rankId++) {
        std::vector<uint32_t> dimIds = {rankId % dimSize[0], rankId / dimSize[0]};
        std::vector<std::vector<RankId>> tempVTopo = {{0, 1}, {0, 2}};
        uint16_t axisId;

        // Die0初始化
        axisId = 0;
        CcuCtxArgAlltoAllMesh2D ctxArg0(dimSize, rankId, axisId, collAlgOp, tempVTopo);
        Eid remoteEid0;
        std::vector<std::shared_ptr<CcuTransport>> transportInstances0;
        std::vector<CcuTransport*> transports0;
        std::vector<uint32_t> cntCke0 = {0, 1, 2, 3};
        for (int i = 0; i < rankSize; i++) {
            if (i != rankId) {
                BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
                LinkData linkData(portType, 0, 1, 0, 1);
                CcuChannelInfo channelInfo;
                vector<CcuJetty*> ccuJettys;
                auto c = std::make_unique<CcuConnection>(
                    linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
                CcuTransport::CclBufferInfo locCclBufInfo;
                c->channelInfo_.channelId = i;
                c->dieId = axisId;
                std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
                t->dieId = axisId;
                t->AppendRes(4, 3);
                t->SetCntCke(cntCke0);
                t->rmtRes.cntCkes = {1128, 1129, 1130, 1131};
                t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
                transportInstances0.push_back(t); // ???
                transports0.push_back(t.get());
            }
        }
        CcuTransportGroup transportGroup0(transports0, 4);
        transportGroup0.cntCkesGroup = {1128, 1129, 1130, 1131};
        CcuContextAlltoAllMesh2D ctx0(ctxArg0, transports0, transportGroup0);

        // Die1初始化
        axisId = 1;
        CcuCtxArgAlltoAllMesh2D ctxArg1(dimSize, rankId, axisId, collAlgOp, tempVTopo);
        Eid remoteEid1;
        std::vector<std::shared_ptr<CcuTransport>> transportInstances1;
        std::vector<CcuTransport*> transports1;
        std::vector<uint32_t> cntCke1 = {0, 1, 2, 3};
        for (int i = 0; i < rankSize; i++) {
            if (i != rankId) {
                BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
                LinkData linkData(portType, 0, 1, 0, 1);
                CcuChannelInfo channelInfo;
                vector<CcuJetty*> ccuJettys;
                auto c = std::make_unique<CcuConnection>(
                    linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
                CcuTransport::CclBufferInfo locCclBufInfo;
                c->channelInfo_.channelId = i;
                c->dieId = axisId;
                std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
                t->dieId = axisId;
                t->AppendRes(4, 3);
                t->SetCntCke(cntCke1);
                t->rmtRes.cntCkes = {1128, 1129, 1130, 1131};
                t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
                transportInstances1.push_back(t); // ???
                transports1.push_back(t.get());
            }
        }
        CcuTransportGroup transportGroup1(transports1, 4);
        transportGroup1.cntCkesGroup = {1128, 1129, 1130, 1131};
        CcuContextAlltoAllMesh2D ctx1(ctxArg1, transports1, transportGroup1);
        HCCL_DEBUG("[CcuContextAlltoAllMesh2D] Die1 Ctx Signature[%s]", ctxArg1.GetCtxSignature().GetData().c_str());

        // 申请资源
        s32 deviceLogicId = 0;
        CcuCtxGroup ctxGroup;
        ctxGroup.ctxs.push_back(std::make_unique<CcuContextAlltoAllMesh2D>(ctx0));
        ctxGroup.ctxs.push_back(std::make_unique<CcuContextAlltoAllMesh2D>(ctx1));
        CcuResPack resPack;
        EXPECT_EQ(CcuCtxMgr::AllocRes(deviceLogicId, ctxGroup, resPack), HCCL_SUCCESS);

        // 注册指令
        InsExeQue::ExtInsExeEntity entity;
        entity.ctxGroup = std::move(ctxGroup);
        InsExeQue::ExtInsExeEntityId entityId = 0;
        EXPECT_EQ(InsExeQue::RegisterExtendInstruction(deviceLogicId, entity, entityId), HCCL_SUCCESS);
    }
}
