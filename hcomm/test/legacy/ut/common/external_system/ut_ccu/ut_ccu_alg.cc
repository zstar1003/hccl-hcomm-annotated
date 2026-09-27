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
#include "rank_table.h"
#include "buffer_type.h"
#include "phy_topo_builder.h"
#define private public
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "ccu_alg_mesh_1D.h"
#include "ccu_alg_mesh_2D.h"
#include "env_config_v2.h"
#include "rdma_handle_manager.h"
#include "ccu_mc2.h"
#include "ccu_mission.h"
#include "ccu_resource.h"
#include "ccu_common_type.h"
#include "communicator_impl.h"
#include "ccu_component.h"
#undef private

using namespace Hccl;
using namespace Ccu;

class CCUAlgorithmTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CCUAlgorithmTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CCUAlgorithmTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CCUAlgorithmTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CCUAlgorithmTest TearDown" << std::endl;
    }

    Resource GetResource()
    {
        Resource resource;

        resource.dieId = 0;
        resource.missionId = 0;

        resource.startCKEId = 0;
        resource.ckeBlockSize = 1024;
        resource.startMSId = 0;
        resource.msBlockSize = 1536;
        resource.startGSAId = 0;
        resource.gsaBlockSize = 3072;
        resource.startXnId = 0;
        resource.xnBlockSize = 3072;
        resource.startLoopEngineId = 0;
        resource.loopEngineBlockSize = 200;
        return resource;
    }
    void InitComm(u32 xDimSize, u32 yDimSize, u32 myRank, CommunicatorImpl& comm)
    {
        LocalRmaBuffer* rmaBuf = nullptr;
        MOCKER_CPP(
            &LocalRmaBufManager::Reg,
            LocalRmaBuffer * (LocalRmaBufManager::*)(std::shared_ptr<CcuBuffer> ccuBuffer, const PortData& portData))
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(rmaBuf));
        comm.rankSize = xDimSize * yDimSize;
        comm.myRank = myRank;
        comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;

        comm.dataBufferManager = std::make_unique<DataBufManager>();
        comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
        comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
        comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
        comm.streamManager = std::make_unique<StreamManager>(&comm);
        comm.socketManager = std::make_unique<SocketManager>(comm, 0, 0, 6000);

        comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
        comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);

        comm.rankGraph = std::make_unique<RankGraph>(0);

        BasePortType hccsPortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);

        std::vector<shared_ptr<ConnInterface>> xDimConnIfceList;
        std::vector<shared_ptr<ConnInterface>> yDimConnIfceList;
        shared_ptr<NetInstance> fabGroup = make_shared<InnerNetInstance>(0, "id");
        comm.rankGraph->AddNetInstance(fabGroup);
        for (u32 i = 0; i < comm.rankSize; i++) {
            auto peer = std::make_shared<NetInstance::Peer>(i, i);
            IpAddress xInputAddr(i);
            IpAddress yInputAddr(i + comm.rankSize);
            shared_ptr<ConnInterface> xSourceIface = std::make_shared<ConnInterface>(
                xInputAddr, AddrPosition::DEVICE, LinkType::PEER2PEER, LinkProtocol::UB_CTP);
            shared_ptr<ConnInterface> ySourceIface = std::make_shared<ConnInterface>(
                yInputAddr, AddrPosition::DEVICE, LinkType::PEER2PEER, LinkProtocol::UB_CTP);
            peer->AddConnInterface(xSourceIface);
            peer->AddConnInterface(ySourceIface);
            peer->AddNetInstance(fabGroup);
            comm.rankGraph->AddPeer(peer);
            fabGroup->AddNode(peer);
            fabGroup->AddRankId(i);
            xDimConnIfceList.push_back(xSourceIface);
            yDimConnIfceList.push_back(ySourceIface);
        }

        for (u32 i = 0; i < comm.rankSize; i++) {
            for (u32 j = 0; j < comm.rankSize; j++) {
                if (i == j) {
                    continue;
                }
                if (i / xDimSize == j / xDimSize) {
                    auto srcPeer = comm.rankGraph->GetPeer(i);
                    auto dstPeer = comm.rankGraph->GetPeer(j);
                    std::shared_ptr<NetInstance::Link> xLink = std::make_shared<NetInstance::Link>(
                        srcPeer, dstPeer, xDimConnIfceList[i], xDimConnIfceList[j], LinkType::PEER2PEER,
                        LinkProtocol::UB_CTP);
                    fabGroup->AddLink(xLink);
                } else if (i % xDimSize == j % xDimSize) {
                    auto srcPeer = comm.rankGraph->GetPeer(i);
                    auto dstPeer = comm.rankGraph->GetPeer(j);
                    std::shared_ptr<NetInstance::Link> yLink = std::make_shared<NetInstance::Link>(
                        srcPeer, dstPeer, yDimConnIfceList[i], yDimConnIfceList[j], LinkType::PEER2PEER,
                        LinkProtocol::UB_CTP);
                    fabGroup->AddLink(yLink);
                }
            }
        }
        comm.rankGraph->InitInnerRanks();
    }
};

static HcclResult DumpCCUInstruction(uint32_t rankId, CCUAlgoMission* mission, uint16_t dieId = 0)
{
    MissionParams sqe = mission->GetMissionParam();
    HCCL_ERROR("==============================Dump Start==============================");
    HCCL_ERROR("rankId = %u, dieId = %u", rankId, dieId);
    HCCL_ERROR("instrStartId = %u", sqe.instStartId);
    HCCL_ERROR("instrCount = %u", sqe.instCnt);
    for (uint32_t i = 0; i < 13; i++) {
        HCCL_ERROR("arg[%u] = %lu", i, sqe.args[i]);
    }
    mission->DumpInstruction();
    HCCL_ERROR("==============================Dump End==============================");

    return HCCL_SUCCESS;
}

TEST_F(CCUAlgorithmTest, CalMultiOpSize)
{
    Resource resource = GetResource();
    std::vector<uint32_t> dimSizeTmp = {1};
    std::unique_ptr<CCUAlgoMission> mission = std::make_unique<AllgatherMesh1D>(0, dimSizeTmp, resource);
    std::vector<uint64_t> args;

    mission->CalMultiOpSize(256 * 1024, args);
    EXPECT_EQ(args.size(), 4);
    EXPECT_EQ(args[0], 262144);
    EXPECT_EQ(args[1], 2147483649);
    EXPECT_EQ(args[2], 0);
    EXPECT_EQ(args[3], 0);

    mission->CalMultiOpSize(256 * 1024 + 4 * 1024, args);
    EXPECT_EQ(args.size(), 4);
    EXPECT_EQ(args[0], 262144);
    EXPECT_EQ(args[1], 2147483649);
    EXPECT_EQ(args[2], 2199023255552);
    EXPECT_EQ(args[3], 4096);

    mission->CalMultiOpSize(256 * 1024 + 4, args);
    EXPECT_EQ(args.size(), 4);
    EXPECT_EQ(args[0], 262144);
    EXPECT_EQ(args[1], 2147483649);
    EXPECT_EQ(args[2], 2199023255552);
    EXPECT_EQ(args[3], 4);

    mission->CalMultiOpSize(256 * 1024 + 4 * 1024 + 4, args);
    EXPECT_EQ(args.size(), 4);
    EXPECT_EQ(args[0], 262144);
    EXPECT_EQ(args[1], 2147483649);
    EXPECT_EQ(args[2], 285873023221760);
    EXPECT_EQ(args[3], 4);
}

TEST_F(CCUAlgorithmTest, CCUAllgather)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 64 * 1024;
    op.dataType = DataType::UINT8;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLGATHER;

    Resource resource = GetResource();
    resource.dieId = 0;
    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(std::make_unique<AllgatherMesh1D>(rankId, dimSizeTmp, resource));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, CCUAllgatherForMC2)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 64 * 1024;
    op.dataType = DataType::UINT8;
    uint32_t turnNum = 4;
    uint32_t tailTurnNum = 1;
    uint32_t tailCount = 64 * 256 * 1024;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLGATHER;

    Resource resource = GetResource();
    resource.dieId = 0;
    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(std::make_unique<MC2AllgatherMesh1D>(rankId, dimSizeTmp, resource));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        static_cast<MC2AllgatherMesh1D*>(mission[rankId].get())
            ->SetTilingData(
                static_cast<uint64_t>(op.inputMem->GetAddr()), static_cast<uint64_t>(op.outputMem->GetAddr()), turnNum,
                op.dataCount, tailTurnNum, tailCount);
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, CCUReduceScatter)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 4096;
    op.dataType = DataType::INT32;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::REDUCESCATTER;
    op.reduceOp = ReduceOp::SUM;

    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
    Resource resource = GetResource();
    resource.dieId = 0;
    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(
            std::make_unique<ReduceScatterMesh1D>(rankId, dimSizeTmp, resource, std::vector<uint16_t>(0), 0));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, MC2CCUReduceScatter)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    std::vector<uint16_t> inputXnIds(8, 2);
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 4096;
    op.dataType = DataType::INT32;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::REDUCESCATTER;
    op.reduceOp = ReduceOp::SUM;

    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
    Resource resource = GetResource();
    resource.dieId = 0;
    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(std::make_unique<ReduceScatterMesh1D>(rankId, dimSizeTmp, resource, inputXnIds, 0, 0, true));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, CCUReduceScatterForMC2)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 4096;
    op.dataType = DataType::INT32;
    uint32_t turnNum = 4;
    uint32_t tailTurnNum = 1;
    uint32_t tailCount = 4096;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::REDUCESCATTER;
    op.reduceOp = ReduceOp::SUM;

    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
    Resource resource = GetResource();
    resource.dieId = 0;
    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(std::make_unique<MC2ReduceScatterMesh1D>(rankId, dimSizeTmp, resource));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        static_cast<MC2ReduceScatterMesh1D*>(mission[rankId].get())
            ->SetTilingData(
                static_cast<uint64_t>(op.inputMem->GetAddr()), static_cast<uint64_t>(op.outputMem->GetAddr()), turnNum,
                op.dataCount, tailTurnNum, tailCount);
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, CCUAllreduce)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 4 * 1024;
    op.dataType = DataType::INT32;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;

    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
    Resource resource = GetResource();
    resource.dieId = 0;
    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(
            std::make_unique<AllreduceMesh1D>(rankId, dimSizeTmp, resource, std::vector<uint16_t>(0), 0, 0));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, CCUAllreduceOneshot)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 4 * 1024;
    op.dataType = DataType::INT32;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;

    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
    Resource resource = GetResource();
    resource.dieId = 0;
    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        auto AllreduceMesh1DPtr
            = std::make_unique<AllreduceMesh1D>(rankId, dimSizeTmp, resource, std::vector<uint16_t>(0), 0, 0);
        AllreduceMesh1DPtr->algoName_ = "one-shot";
        mission.push_back(std::move(AllreduceMesh1DPtr));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, MC2CCUAllreduce)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    std::vector<uint16_t> inputXnIds(8, 2);
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 4 * 1024;
    op.dataType = DataType::INT32;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;

    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
    Resource resource = GetResource();
    resource.dieId = 0;
    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(std::make_unique<AllreduceMesh1D>(rankId, dimSizeTmp, resource, inputXnIds, 0, 0, true));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, CCUAlltoAll)
{
    std::vector<uint32_t> dimSizeTmp = {2};
    uint16_t rankSize = dimSizeTmp[0];
    CollOperator op;

    op.dataCount = 64 * 1024 * rankSize;
    op.dataType = DataType::UINT8;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLTOALL;

    std::vector<std::unique_ptr<CCUAlgoMission>> mission;
    std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
    Resource resource = GetResource();
    resource.dieId = 0;
    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission.push_back(std::make_unique<AlltoAllMesh1D>(rankId, dimSizeTmp, resource));
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GetExchangeResource(exchangeResource[rankId]);
    }

    std::vector<CCUExchangeResource> tmp;

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        tmp.clear();
        for (uint16_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
            tmp.push_back(exchangeResource[srcRankId][rankId]);
        }

        mission[rankId]->SetExchangeResource(tmp);
    }

    for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
        mission[rankId]->GeneInstruction(op);
        mission[rankId]->GeneArgs(op);

        DumpCCUInstruction(rankId, mission[rankId].get());
    }
}

TEST_F(CCUAlgorithmTest, CCU2DFullMeshReduceScatter)
{
    std::vector<uint32_t> dimSizeTmp = {2, 2};
    uint32_t localSize = dimSizeTmp[0];
    uint16_t rankSize = dimSizeTmp[0] * dimSizeTmp[1];
    CollOperator op;

    op.dataCount = 64 * 1024;
    op.dataType = DataType::UINT8;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::REDUCESCATTER;
    op.reduceOp = ReduceOp::SUM;

    Resource resource0 = GetResource();
    Resource resource1 = GetResource();
    resource0.dieId = 0;
    resource1.dieId = 1;

    std::vector<std::vector<std::unique_ptr<CCUAlgoMission>>> mission;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        std::vector<std::unique_ptr<CCUAlgoMission>> rankMission;
        rankMission.push_back(
            std::make_unique<ReduceScatterMesh2D>(rankId, dimSizeTmp, resource0, std::vector<uint16_t>(0), 0));
        rankMission.push_back(
            std::make_unique<ReduceScatterMesh2D>(rankId, dimSizeTmp, resource1, std::vector<uint16_t>(0), 0));
        mission.push_back(std::move(rankMission));
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GetExchangeResource(exchangeResource[rankId]);
        }
        std::vector<CCUExchangeResource> tmp;
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            tmp.clear();
            for (uint32_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
                uint32_t dimId = dieId == 0 ? rankId % localSize : rankId / localSize;
                if (dieId == 0 ? (rankId / localSize == srcRankId / localSize) :
                                 (rankId % localSize == srcRankId % localSize)) {
                    tmp.push_back(exchangeResource[srcRankId][dimId]);
                }
            }
            mission[rankId][dieId]->SetExchangeResource(tmp);
        }
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GeneInstruction(op);
            mission[rankId][dieId]->GeneArgs(op);
            DumpCCUInstruction(rankId, mission[rankId][dieId].get(), dieId);
        }
    }
}

TEST_F(CCUAlgorithmTest, CCU2DFullMeshAllReduce)
{
    std::vector<uint32_t> dimSizeTmp = {2, 2};
    uint32_t localSize = dimSizeTmp[0];
    uint16_t rankSize = dimSizeTmp[0] * dimSizeTmp[1];
    CollOperator op;

    op.dataCount = 64 * 1024;
    op.dataType = DataType::UINT8;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.scratchMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;

    Resource resource0 = GetResource();
    Resource resource1 = GetResource();
    resource0.dieId = 0;
    resource1.dieId = 1;

    std::vector<std::vector<std::unique_ptr<CCUAlgoMission>>> mission;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        std::vector<std::unique_ptr<CCUAlgoMission>> rankMission;
        rankMission.push_back(
            std::make_unique<AllreduceMesh2DOneShot>(rankId, dimSizeTmp, resource0, std::vector<uint16_t>(0), 0));
        rankMission.push_back(
            std::make_unique<AllreduceMesh2DOneShot>(rankId, dimSizeTmp, resource1, std::vector<uint16_t>(0), 0));
        mission.push_back(std::move(rankMission));
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GetExchangeResource(exchangeResource[rankId]);
        }
        std::vector<CCUExchangeResource> tmp;
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            tmp.clear();
            for (uint32_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
                uint32_t dimId = dieId == 0 ? rankId % localSize : rankId / localSize;
                if (dieId == 0 ? (rankId / localSize == srcRankId / localSize) :
                                 (rankId % localSize == srcRankId % localSize)) {
                    tmp.push_back(exchangeResource[srcRankId][dimId]);
                }
            }
            mission[rankId][dieId]->SetExchangeResource(tmp);
        }
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GeneInstruction(op);
            mission[rankId][dieId]->GeneArgs(op);
            DumpCCUInstruction(rankId, mission[rankId][dieId].get(), dieId);
        }
    }
}

TEST_F(CCUAlgorithmTest, CCU2DFullMeshAllreduceTwoShot)
{
    std::vector<uint32_t> dimSizeTmp = {2, 2};
    uint32_t localSize = dimSizeTmp[0];
    uint16_t rankSize = dimSizeTmp[0] * dimSizeTmp[1];
    CollOperator op;

    op.dataCount = 64 * 1024;
    op.dataType = DataType::UINT8;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.scratchMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;

    Resource resource0 = GetResource();
    Resource resource1 = GetResource();
    resource0.dieId = 0;
    resource1.dieId = 1;

    std::vector<std::vector<std::unique_ptr<CCUAlgoMission>>> mission;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        std::vector<std::unique_ptr<CCUAlgoMission>> rankMission;
        rankMission.push_back(
            std::make_unique<AllreduceMesh2DTwoShot>(rankId, dimSizeTmp, resource0, std::vector<uint16_t>(0), 0));
        rankMission.push_back(
            std::make_unique<AllreduceMesh2DTwoShot>(rankId, dimSizeTmp, resource1, std::vector<uint16_t>(0), 0));
        mission.push_back(std::move(rankMission));
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GetExchangeResource(exchangeResource[rankId]);
        }
        std::vector<CCUExchangeResource> tmp;
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            tmp.clear();
            for (uint32_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
                uint32_t dimId = dieId == 0 ? rankId % localSize : rankId / localSize;
                if (dieId == 0 ? (rankId / localSize == srcRankId / localSize) :
                                 (rankId % localSize == srcRankId % localSize)) {
                    tmp.push_back(exchangeResource[srcRankId][dimId]);
                }
            }
            mission[rankId][dieId]->SetExchangeResource(tmp);
        }
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GeneInstruction(op);
            mission[rankId][dieId]->GeneArgs(op);
            DumpCCUInstruction(rankId, mission[rankId][dieId].get(), dieId);
        }
    }
}

TEST_F(CCUAlgorithmTest, CCU2DFullMeshAllGather)
{
    std::vector<uint32_t> dimSizeTmp = {2, 2};
    uint32_t localSize = dimSizeTmp[0];
    uint16_t rankSize = dimSizeTmp[0] * dimSizeTmp[1];
    CollOperator op;

    op.dataCount = 64 * 1024;
    op.dataType = DataType::UINT8;

    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;

    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.scratchMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLGATHER;

    Resource resource0 = GetResource();
    Resource resource1 = GetResource();
    resource0.dieId = 0;
    resource1.dieId = 1;

    std::vector<std::vector<std::unique_ptr<CCUAlgoMission>>> mission;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        std::vector<std::unique_ptr<CCUAlgoMission>> rankMission;
        rankMission.push_back(
            std::make_unique<AllGatherMesh2D>(rankId, dimSizeTmp, resource0, std::vector<uint16_t>(0), 0));
        rankMission.push_back(
            std::make_unique<AllGatherMesh2D>(rankId, dimSizeTmp, resource1, std::vector<uint16_t>(0), 0));
        mission.push_back(std::move(rankMission));
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        std::vector<std::vector<CCUExchangeResource>> exchangeResource(rankSize);
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GetExchangeResource(exchangeResource[rankId]);
        }
        std::vector<CCUExchangeResource> tmp;
        for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
            tmp.clear();
            for (uint32_t srcRankId = 0; srcRankId < rankSize; srcRankId++) {
                uint32_t dimId = dieId == 0 ? rankId % localSize : rankId / localSize;
                if (dieId == 0 ? (rankId / localSize == srcRankId / localSize) :
                                 (rankId % localSize == srcRankId % localSize)) {
                    tmp.push_back(exchangeResource[srcRankId][dimId]);
                }
            }
            mission[rankId][dieId]->SetExchangeResource(tmp);
        }
    }
    for (uint16_t dieId = 0; dieId < 2; dieId++) {
        for (uint16_t rankId = 0; rankId < rankSize; rankId++) {
            mission[rankId][dieId]->GeneInstruction(op);
            mission[rankId][dieId]->GeneArgs(op);
            DumpCCUInstruction(rankId, mission[rankId][dieId].get(), dieId);
        }
    }
}

TEST_F(CCUAlgorithmTest, CCUGetMissionParams1D)
{
    u32 xDimSize = 2;
    u32 yDimSize = 1;
    u32 myRank = 0;
    CommunicatorImpl comm;
    InitComm(xDimSize, yDimSize, myRank, comm);
    void* rdmaHandle = (void*)0x100;
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    MOCKER(HraGetDieAndFuncId).stubs().with(mockcpp::any()).will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
    GenTopoFile();
    PhyTopoBuilder phyTopoBuilder;
    phyTopoBuilder.Build("topo.json");

    CcuComponent ccuComponent(&comm);

    CollOperator op;
    op.dataCount = 64 * 1024;
    op.dataType = DataType::UINT8;
    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;
    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.scratchMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::REDUCESCATTER;
    op.reduceOp = ReduceOp::SUM;

    ccuComponent.GetMissionParams1D(op);
}

TEST_F(CCUAlgorithmTest, CCUGetMissionParams2D_RS)
{
    void* ccuConfigPtr1 = (void*)(&(EnvConfig::GetInstance().GetCcuConfig()));
    EnvCcuConfig* ccuConfigPtr2 = (EnvCcuConfig*)(ccuConfigPtr1);
    ccuConfigPtr2->ioDieNum.value = 2;
    ResourceManager::GetInstance().InitResourceManager();
    MOCKER_CPP(&CcuComponent::CreateLocalOpChannel).stubs().will(ignoreReturnValue());

    u32 xDimSize = 2;
    u32 yDimSize = 2;
    u32 myRank = 0;
    CommunicatorImpl comm;
    InitComm(xDimSize, yDimSize, myRank, comm);
    void* rdmaHandle = (void*)0x100;
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    MOCKER(HraGetDieAndFuncId).stubs().with(mockcpp::any()).will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));

    CcuComponent ccuComponent(&comm);

    ccuComponent.innerDieChannelInfoMap[0] = ChannelInfo();
    ccuComponent.innerDieChannelInfoMap[1] = ChannelInfo();
    ccuComponent.interDieChannelInfoMap[0] = ChannelInfo();
    ccuComponent.interDieChannelInfoMap[1] = ChannelInfo();
    CollOperator op;
    op.dataCount = 64 * 1024;
    op.dataType = DataType::FP16;
    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;
    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.scratchMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::REDUCESCATTER;
    op.reduceOp = ReduceOp::SUM;

    ccuComponent.GetMissionParams2D(op);
    ccuConfigPtr2->ioDieNum.value = 1;
    EXPECT_NO_THROW(ResourceManager::GetInstance().InitResourceManager());
}

TEST_F(CCUAlgorithmTest, CCUGetMissionParams2D_AG)
{
    void* ccuConfigPtr1 = (void*)(&(EnvConfig::GetInstance().GetCcuConfig()));
    EnvCcuConfig* ccuConfigPtr2 = (EnvCcuConfig*)(ccuConfigPtr1);
    ccuConfigPtr2->ioDieNum.value = 2;
    ResourceManager::GetInstance().InitResourceManager();
    MOCKER_CPP(&CcuComponent::CreateLocalOpChannel).stubs().will(ignoreReturnValue());

    u32 xDimSize = 2;
    u32 yDimSize = 2;
    u32 myRank = 0;
    CommunicatorImpl comm;
    InitComm(xDimSize, yDimSize, myRank, comm);
    void* rdmaHandle = (void*)0x100;
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    MOCKER(HraGetDieAndFuncId).stubs().with(mockcpp::any()).will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));

    CcuComponent ccuComponent(&comm);

    ccuComponent.innerDieChannelInfoMap[0] = ChannelInfo();
    ccuComponent.innerDieChannelInfoMap[1] = ChannelInfo();
    ccuComponent.interDieChannelInfoMap[0] = ChannelInfo();
    ccuComponent.interDieChannelInfoMap[1] = ChannelInfo();
    CollOperator op;
    op.dataCount = 64 * 1024;
    op.dataType = DataType::INT32;
    uint64_t size = DataTypeSizeGet(op.dataType) * op.dataCount;
    op.inputMem = DevBuffer::Create(0x100, size);
    op.outputMem = DevBuffer::Create(0x100, size);
    op.scratchMem = DevBuffer::Create(0x100, size);
    op.opType = OpType::ALLGATHER;

    ccuComponent.GetMissionParams2D(op);
    ccuConfigPtr2->ioDieNum.value = 1;
    EXPECT_NO_THROW(ResourceManager::GetInstance().InitResourceManager());
}

TEST_F(CCUAlgorithmTest, CheckCcuDataType_ShouldThrowException_HighPrecisionReduce)
{
    CollOperator op;
    op.dataType = DataType::UINT16;
    op.outputDataType = DataType::UINT16;

    op.opType = OpType::REDUCESCATTER;
    EXPECT_THROW(CheckCcuDataType(op), CcuApiException);
    op.opType = OpType::ALLREDUCE;
    EXPECT_THROW(CheckCcuDataType(op), CcuApiException);

    op.dataType = DataType::INT32;
    op.outputDataType = DataType::FP8E5M2;

    op.opType = OpType::REDUCESCATTER;
    EXPECT_THROW(CheckCcuDataType(op), CcuApiException);
    op.opType = OpType::ALLREDUCE;
    EXPECT_THROW(CheckCcuDataType(op), CcuApiException);

    op.dataType = DataType::INT8;
    op.outputDataType = DataType::HIF8;

    op.opType = OpType::REDUCESCATTER;
    EXPECT_THROW(CheckCcuDataType(op), CcuApiException);
    op.opType = OpType::ALLREDUCE;
    EXPECT_THROW(CheckCcuDataType(op), CcuApiException);
}

TEST_F(CCUAlgorithmTest, CheckCcuDataType_ShouldNotThrowException_HighPrecisionReduce)
{
    CollOperator op;
    op.dataType = DataType::FP32;
    op.outputDataType = DataType::FP32;

    op.opType = OpType::REDUCESCATTER;
    EXPECT_NO_THROW(CheckCcuDataType(op));
    op.opType = OpType::ALLREDUCE;
    EXPECT_NO_THROW(CheckCcuDataType(op));

    op.dataType = DataType::INT16;
    op.outputDataType = DataType::INT16;

    op.opType = OpType::REDUCESCATTER;
    EXPECT_NO_THROW(CheckCcuDataType(op));
    op.opType = OpType::ALLREDUCE;
    EXPECT_NO_THROW(CheckCcuDataType(op));

    op.dataType = DataType::HIF8;
    op.outputDataType = DataType::FP16;

    op.opType = OpType::REDUCESCATTER;
    EXPECT_NO_THROW(CheckCcuDataType(op));
    op.opType = OpType::ALLREDUCE;
    EXPECT_NO_THROW(CheckCcuDataType(op));

    op.dataType = DataType::FP8E5M2;
    op.outputDataType = DataType::FP32;

    op.opType = OpType::REDUCESCATTER;
    EXPECT_NO_THROW(CheckCcuDataType(op));
    op.opType = OpType::ALLREDUCE;
    EXPECT_NO_THROW(CheckCcuDataType(op));
}

TEST_F(CCUAlgorithmTest, CheckCcuDataType_ShouldThrowException_WhenInconsistentDataType)
{
    CollOperator op;
    op.dataType = DataType::FP32;
    op.outputDataType = DataType::FP16;

    op.opType = OpType::BROADCAST;
    EXPECT_THROW(CheckCcuDataType(op), CcuApiException);
}

TEST_F(CCUAlgorithmTest, CheckCcuDataType_ShouldNotThrowException_WhenConsistentDataType)
{
    CollOperator op;
    op.dataType = DataType::FP32;
    op.outputDataType = DataType::FP32;

    op.opType = OpType::BROADCAST;
    EXPECT_NO_THROW(CheckCcuDataType(op));

    op.dataType = DataType::FP8E5M2;
    op.outputDataType = DataType::FP8E5M2;

    op.opType = OpType::BROADCAST;
    EXPECT_NO_THROW(CheckCcuDataType(op));
}
