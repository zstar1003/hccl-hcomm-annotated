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
#define private public
#define protected public
#include <memory>
#include <vector>

#include <chrono>
#include <iostream>

#include "ccu_transport.h"
#include "ccu_transport_group.h"
#include "ccu_transport_manager.h"
#include "ccu_rep.h"
#include "ccu_ctx.h"
#include "ccu_rep_translator.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "log.h"
#include "ccu_context_common.h"
#include "ccu_instruction_all_reduce_mesh1d.h"
#include "ccu_rep_context.h"
#include "ccu_rep_loopblock.h"
#include "ccu_instruction_all_to_all_mesh2d.h"
#include "common_interface.h"

#undef private

using namespace Hccl;
using namespace CcuRep;

class CcuDfxTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuDfxTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuDfxTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        JfcHandle jfcHandle = 1;
        MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
        std::cout << "A Test case in CcuDfxTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuDfxTest TearDown" << std::endl;
    }
};

class CcuProfilingDfxTest : public CcuContext {
public:
    CcuProfilingDfxTest(
        CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

    void GroupBroadcast_expection_test()
    {
        uint32_t size = 8;
        uint32_t id = 0;
        std::vector<Variable> input;
        std::vector<Variable> output;
        std::vector<Variable> token;
        for (uint32_t i = 0; i < size; i++) {
            input.emplace_back(CreateVariable());
            output.emplace_back(CreateVariable());
            token.emplace_back(CreateVariable());
        }
        Variable offset = CreateVariable();
        GroupOpSize goSize = CreateGroupOpSize();
        CcuRep::Variable sliceSize = CreateVariable();
        Memory src = CreateMemory();
        std::vector<Memory> dst;
        for (uint32_t i = 0; i < size; i++) {
            dst.emplace_back(CreateMemory());
        }
        uint16_t selfBit = 1 << id;
        uint16_t allBit = ((1 << size) - 1) & (~(1 << id));
        Load(input[id]);
        Load(output[id]);
        Load(token[id]);
        Load(sliceSize);
        Load(goSize);
        for (auto t : transports) {
            WriteVariableWithSignal(*t, input[id], 0, 0, selfBit);  // index = 0，传递input信息
            WriteVariableWithSignal(*t, output[id], 1, 1, selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(*t, token[id], 2, 2, selfBit);  // index = 2，传递token信息
        }
        src.addr = input[id];
        src.token = token[id];
        uint32_t dstId = 0;
        uint32_t curId = 0;
        for (uint32_t r = 0; r < size; r++) {
            if (r != id) {
                curId = dstId;
                dstId++;
            } else {
                curId = size - 1;
            }
            dst[curId].addr = output[r];
            dst[curId].addr += offset;
            dst[curId].token = token[r];
        }
        GroupBroadcast(transports, dst, src, goSize);
    }
    void GroupReduce_test()
    {
        uint32_t size = 8;
        uint32_t id = 0;
        std::vector<Variable> input;
        std::vector<Variable> output;
        std::vector<Variable> token;
        for (uint32_t i = 0; i < size; i++) {
            input.emplace_back(CreateVariable());
            output.emplace_back(CreateVariable());
            token.emplace_back(CreateVariable());
        }
        Variable offset = CreateVariable();
        GroupOpSize goSize = CreateGroupOpSize();
        CcuRep::Variable sliceSize = CreateVariable();
        std::vector<Memory> src;
        Memory dst = CreateMemory();
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateMemory());
        }
        uint16_t selfBit = 1 << id;
        uint16_t allBit = ((1 << size) - 1) & (~(1 << id));

        Load(input[id]);
        Load(output[id]);
        Load(token[id]);
        Load(sliceSize);
        Load(goSize);

        for (auto t : transports) {
            WriteVariableWithSignal(*t, input[id], 0, 0, selfBit);  // index = 0，传递input信息
            WriteVariableWithSignal(*t, output[id], 1, 1, selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(*t, token[id], 2, 2, selfBit);  // index = 2，传递token信息
        }
        uint32_t dstId = 0;
        uint32_t curId = 0;
        for (uint32_t r = 0; r < size; r++) {
            if (r != id) {
                curId = dstId;
                dstId++;
            } else {
                curId = size - 1;
            }
            src[curId].addr = input[r];
            src[curId].addr += offset;
            src[curId].token = token[r];
        }
        dst.addr = output[id];
        dst.token = token[id];

        GroupReduce(transports, dst, src, goSize, DataType::FP32, DataType::FP32, ReduceOp::MAX);
    }

protected:
    void Algorithm() override { return; }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        const CcuTaskArgAlltoAllMesh2D* taskArg = dynamic_cast<const CcuTaskArgAlltoAllMesh2D*>(&arg);
        if (taskArg == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAlltoAllMesh2D::taskArg ptr is null"));
        }

        // input&output&buffer地址
        uint64_t inputAddr = taskArg->inputAddr;
        uint64_t outputAddr = taskArg->outputAddr;
        uint64_t scratchAddr = taskArg->scratchAddr;
        uint64_t tokenValue = taskArg->token;
        uint64_t sliceSizeValue = taskArg->aSize + taskArg->bSize;

        // scratch的前rankSize*sliceSize大小为bufferY，后一块为bufferX
        // die0第一轮写到对端的bufferY，第二轮从本端bufferX发送；die1第一轮写到对端的bufferX，第二轮从本端bufferY发送
        uint64_t bufferAAddr = 0;
        uint64_t bufferBAddr = 0;
        bufferAAddr = scratchAddr + sliceSizeValue; // bufferX
        bufferBAddr = scratchAddr;                  // 不需要交换给对端，是bufferY

        // loopgroup按照sliceSize大小做本地搬运，只die0执行
        auto goSize = CalGoSize(taskArg->aSize + taskArg->bSize);
        std::cout << "offset = " << goSize[0] << ", loopIterNum = " << goSize[1] << ", parallelParam= " << goSize[2]
                  << ", tailSize= " << goSize[3] << std::endl;
        return {
            inputAddr, outputAddr, tokenValue, sliceSizeValue, goSize[0], goSize[1], goSize[2], goSize[3],
        };
    }
};

class ccuTaskArgTest {
public:
    void Init(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t sendStride, uint64_t recvStride,
        uint64_t sendLength, uint64_t aSize, uint64_t bSize, uint64_t baseOffset, uint64_t token)
    {
        inputAddr_ = inputAddr;
        outputAddr_ = outputAddr;
        scratchAddr_ = scratchAddr;
        sendStride_ = sendStride;
        recvStride_ = recvStride;
        sendLength_ = sendLength;
        aSize_ = aSize;
        bSize_ = bSize;
        baseOffset_ = baseOffset;
        token_ = token;
        return;
    }
    std::unique_ptr<CcuTaskArg> GetTaskArg()
    {
        return std::make_unique<CcuTaskArgAlltoAllMesh2D>(
            inputAddr_, outputAddr_, scratchAddr_, sendStride_, recvStride_, sendLength_, aSize_, bSize_, baseOffset_,
            token_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t sendStride_;
    uint64_t recvStride_;
    uint64_t sendLength_; // 多轮时的单个数据块总大小
    uint64_t aSize_;
    uint64_t bSize_;
    uint64_t baseOffset_;
    uint64_t token_;
};

std::vector<CcuProfilingInfo> createCcuGroupBroadcastTest(uint32_t asize, uint32_t bsize)
{
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23 + i;
            c->dieId = 1;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.SetMissionId(1);
    ctx.GroupBroadcast_expection_test();
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    return ccuprofilinginfo;
}

TEST_F(CcuDfxTest, localwait_block_test)
{
    auto loopBlock = std::make_shared<CcuRepLoopBlock>("la");
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    uint32_t ckeId = 0;
    uint32_t mask = 4;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23 + i;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.SetCurrentBlock(loopBlock);
    MaskSignal maskSignalInstance;
    maskSignalInstance.Reset(ckeId);
    ctx.LocalWait(maskSignalInstance, mask);
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 1);
}

TEST_F(CcuDfxTest, localwait_test)
{
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    uint32_t ckeId = 1;
    uint32_t mask = 4;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23 + i;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    MaskSignal maskSignalInstance;
    maskSignalInstance.Reset(ckeId);
    ctx.LocalWait(maskSignalInstance, mask);
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 2);
    EXPECT_EQ(ccuprofilinginfo[1].name, "LocalWait");
    EXPECT_EQ(ccuprofilinginfo[1].ckeId, 1);
    EXPECT_EQ(ccuprofilinginfo[1].mask, 4);
}

TEST_F(CcuDfxTest, RemoteWait_block_test)
{
    auto loopBlock = std::make_shared<CcuRepLoopBlock>("la");
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23 + i;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.SetCurrentBlock(loopBlock);
    ctx.RemoteWait(*transports[0], 0, 1);

    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 1);
}

TEST_F(CcuDfxTest, RemoteWait_test)
{
    uint32_t rankId = 0;
    uint32_t rankSize = 8;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23 + i;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.SetMissionId(6);
    ctx.RemoteWait(*transports[0], 2, 1);
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 2);
    EXPECT_EQ(ccuprofilinginfo[1].name, "RemoteWait");
    EXPECT_EQ(ccuprofilinginfo[1].type, 1);
    EXPECT_EQ(ccuprofilinginfo[1].mask, 1);
}
TEST_F(CcuDfxTest, GroupWait_block_Test)
{
    auto loopBlock = std::make_shared<CcuRepLoopBlock>("la");
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23 + i;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.SetCurrentBlock(loopBlock);
    ctx.GroupWait(transportGroup, 0, 1);
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 1);
}

TEST_F(CcuDfxTest, GroupWait_Test)
{
    uint32_t rankId = 1;
    uint32_t rankSize = 8;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23 + i;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.GroupWait(transportGroup, 1, 1);
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 2);
    EXPECT_EQ(ccuprofilinginfo[1].name, "GroupWait");
    EXPECT_EQ(ccuprofilinginfo[1].type, 1);
    EXPECT_EQ(ccuprofilinginfo[1].ckeId, 129);
    EXPECT_EQ(ccuprofilinginfo[1].mask, 1);
}

TEST_F(CcuDfxTest, CcuContextProfiling_Perform_multiple_tasks)
{
    uint32_t rankId = 1;
    uint32_t rankSize = 8;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    uint32_t ckeId = 2;
    MaskSignal maskSignalInstance;
    maskSignalInstance.Reset(ckeId);
    ctx.LocalWait(maskSignalInstance, 4);
    ctx.RemoteWait(*transports[0], 1, 1);
    ctx.GroupWait(transportGroup, 1, 1);
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 4);
}
TEST_F(CcuDfxTest, CcuContextProfilingTest_Test)
{
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};
    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    uint32_t ckeId = 2;
    MaskSignal maskSignalInstance;
    maskSignalInstance.Reset(ckeId);
    for (int i = 0; i < 30; i++) {
        ctx.LocalWait(maskSignalInstance, 4);
        ctx.RemoteWait(*transports[0], 1, 1);
        ctx.GroupWait(transportGroup, 1, 1);
    }
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 91);
}
TEST_F(CcuDfxTest, GroupBroadcastTest)
{
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};

    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.GroupBroadcast_expection_test();
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    DumpCcuProfilingInfo(ccuprofilinginfo);
    EXPECT_EQ(ccuprofilinginfo.size(), 2);
    EXPECT_EQ(ccuprofilinginfo[1].name, "GroupBroadcast");
    EXPECT_EQ(ccuprofilinginfo[1].channelId[0], 23);
}

TEST_F(CcuDfxTest, GroupReduceTest)
{
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            CcuChannelInfo channelInfo;
            vector<CcuJetty*> ccuJettys;
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
            c->channelInfo_.channelId = 23;
            CcuTransport::CclBufferInfo locCclBufInfo;
            std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
            t->AppendRes(3, 3);
            t->SetCntCke(cntCke);
            t->rmtRes.cntCkes = {128, 129, 130};
            t->rmtRes.xns = {1024 + rankId, 1024 + rankSize + rankId, 1024 + rankSize * 2 + rankId};
            transportInstances.push_back(t);
            transports.push_back(t.get());
        }
    }
    CcuTransportGroup transportGroup(transports, 3);
    transportGroup.cntCkesGroup = {128, 129, 130};

    CcuProfilingDfxTest ctx(ctxArg, transports, transportGroup);
    ctx.SetMissionId(1);
    ctx.SetInstrId(1);
    ctx.GroupReduce_test();
    ccuTaskArgTest taskArg;
    taskArg.Init(5, 6, 7, 8, 9, 10, asize, bsize, 13, 14);
    std::vector<CcuProfilingInfo> ccuprofilinginfo;
    auto ret = ctx.GetCcuProfilingInfo(*taskArg.GetTaskArg(), ccuprofilinginfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GetCcuProfilingInfo is failed!");
    }
    EXPECT_EQ(ccuprofilinginfo.size(), 2);
    EXPECT_EQ(ccuprofilinginfo[1].name, "GroupReduce");
    EXPECT_EQ(ccuprofilinginfo[1].channelId[0], 23);
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_0)
{
    uint32_t asize = 0;
    uint32_t bsize = 0;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 1);
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_1k)
{
    uint32_t asize = 1024;
    uint32_t bsize = 0;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_4k)
{
    uint32_t asize = 1024;
    uint32_t bsize = 3 * 1024;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_5k)
{
    uint32_t asize = 2 * 1024;
    uint32_t bsize = 3 * 1024;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_255k)
{
    uint32_t asize = 252 * 1024;
    uint32_t bsize = 3 * 1024;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_256k)
{
    uint32_t asize = 252 * 1024;
    uint32_t bsize = 4 * 1024;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_256k_1)
{
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 - 1;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_256k__1)
{
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 1024 + 1;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_260k)
{
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 5 * 1024;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_260k__2)
{
    uint32_t asize = 255 * 1024;
    uint32_t bsize = 5 * 1024 + 2;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_300k)
{
    uint32_t asize = 299 * 1024;
    uint32_t bsize = 1024;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 2);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}

TEST_F(CcuDfxTest, GetProfilingInfoTest_total_size_equals_699k__1)
{
    uint32_t asize = 600 * 1024 + 1;
    uint32_t bsize = 99 * 1024;
    std::vector<CcuProfilingInfo> profilingInfo = createCcuGroupBroadcastTest(asize, bsize);
    EXPECT_EQ(profilingInfo.size(), 3);
    EXPECT_EQ(profilingInfo[1].name, "GroupBroadcast");
}
