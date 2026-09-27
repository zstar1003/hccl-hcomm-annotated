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

#include <memory>
#include <vector>

#include <chrono>
#include <iostream>

#define private public
#define protected public
#include "ccu_transport.h"
#include "ccu_transport_group.h"
#undef private
#undef protected

#include "ccu_rep.h"
#include "ccu_ctx.h"
#include "ccu_rep_translator.h"

#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"

#include "log.h"

using namespace Hccl;
using namespace CcuRep;

class CcuContextTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuContextTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuContextTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        JfcHandle jfcHandle = 1;
        MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
        std::cout << "A Test case in CcuContextTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuContextTest TearDown" << std::endl;
    }
};

class CcuCtxArgTest : public CcuCtxArg {
public:
    explicit CcuCtxArgTest(uint32_t rankId, uint32_t rankSize) : rankId(rankId), rankSize(rankSize) {}
    virtual ~CcuCtxArgTest() = default;
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        signature.Append("Test");
        return signature;
    }
    uint32_t rankId;
    uint32_t rankSize;
};

class CcuTaskArgTest : public CcuTaskArg {
public:
    explicit CcuTaskArgTest(uint64_t inputAddr, uint64_t outputAddr, uint64_t size)
        : inputAddr(inputAddr),
          outputAddr(outputAddr),
          size(size)
    {}
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t size;
};

class CcuContextAG : public CcuContext {
public:
    CcuContextAG(CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {
        id = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankId;
        size = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankSize;
    }

protected:
    void Algorithm() override
    {
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

        Memory src = CreateMemory();
        std::vector<Memory> dst;
        for (uint32_t i = 0; i < size; i++) {
            dst.emplace_back(CreateMemory());
        }

        uint16_t selfBit = 1 << id;
        uint16_t allBit = ((1 << size) - 1) & (~(1 << id));

        Load(input[id]);
        Load(output[id]);
        Load(offset);
        Load(goSize);
        Load(token[id]);

        for (auto t : transports) {
            WriteVariableWithSignal(*t, input[id], 0, 0, selfBit);  // index = 0，传递input信息
            WriteVariableWithSignal(*t, output[id], 1, 1, selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(*t, token[id], 2, 2, selfBit);  // index = 2，传递token信息
        }
        GroupWait(*transportGroup, 0, allBit); // index = 0，传递input信息
        GroupWait(*transportGroup, 1, allBit); // index = 1，传递output信息
        GroupWait(*transportGroup, 2, allBit); // index = 2，传递token信息

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

        for (auto t : transports) {
            RemotePost(*t, 0, selfBit);
        }
        GroupWait(*transportGroup, 0, allBit);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        auto taskArg = dynamic_cast<const CcuTaskArgTest*>(&arg);
        auto goSize = CalGoSize(taskArg->size);

        return {taskArg->inputAddr, taskArg->outputAddr, 0, goSize[0], goSize[1], goSize[2], goSize[3], 0};
    }

private:
    uint32_t id;
    uint32_t size;
};

class CcuContextRS : public CcuContext {
public:
    CcuContextRS(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {
        id = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankId;
        size = dynamic_cast<const CcuCtxArgTest*>(&arg)->rankSize;
    }

protected:
    void Algorithm() override
    {
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

        std::vector<Memory> src;
        Memory dst = CreateMemory();
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateMemory());
        }

        uint16_t selfBit = 1 << id;
        uint16_t allBit = ((1 << size) - 1) & (~(1 << id));

        Load(input[id]);
        Load(output[id]);
        Load(offset);
        Load(goSize);
        Load(token[id]);

        for (auto t : transports) {
            WriteVariableWithSignal(*t, input[id], 0, 0, selfBit);  // index = 0，传递input信息
            WriteVariableWithSignal(*t, output[id], 1, 1, selfBit); // index = 1，传递output信息
            WriteVariableWithSignal(*t, token[id], 2, 2, selfBit);  // index = 2，传递token信息
        }
        GroupWait(*transportGroup, 0, allBit); // index = 0，传递input信息
        GroupWait(*transportGroup, 1, allBit); // index = 1，传递output信息
        GroupWait(*transportGroup, 2, allBit); // index = 2，传递token信息

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

        GroupReduce(transports, dst, src, goSize, DataType::FP32, DataType::FP32, ReduceOp::SUM);

        for (const auto& t : transports) {
            RemotePost(*t, 0, selfBit);
        }
        GroupWait(*transportGroup, 0, allBit);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        auto taskArg = dynamic_cast<const CcuTaskArgTest*>(&arg);
        auto goSize = CalGoSize(taskArg->size);

        return {taskArg->inputAddr, taskArg->outputAddr, 0, goSize[0], goSize[1], goSize[2], goSize[3], 0};
    }

private:
    uint32_t id;
    uint32_t size;
};

class CcuContextTestMultiArgs : public CcuContext {
public:
    CcuContextTestMultiArgs(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        std::vector<Variable> input(14);
        for (uint32_t i = 0; i < 14; i++) {
            input.emplace_back(CreateVariable());
        }
        for (uint32_t i = 0; i < 14; i++) {
            Load(input[i]);
        }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        std::vector<uint64_t> args(14);
        for (int i = 0; i < 14; i++) {
            args[i] = i;
        }
        return args;
    }
};

class CcuContextTestTracker : public CcuContext {
public:
    CcuContextTestTracker(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup),
          var(CreateVariable())
    {}

protected:
    void Algorithm() override {}
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }

private:
    Variable var;
};

class CcuContextTestVariable : public CcuContext {
public:
    CcuContextTestVariable(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        Variable a = CreateVariable();
        Variable b = CreateVariable();
        Variable c = CreateVariable();
        a = b + c;

        Address aa = CreateAddress();
        Address ab = CreateAddress();
        Address ac = CreateAddress();
        aa = ab + ac;

        aa = ab;
        aa = 0;
        aa = a + ab;
        aa = ab + a;
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

class CcuContextTestDataTransfer : public CcuContext {
public:
    CcuContextTestDataTransfer(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        CcuRep::CcuBuffer buf = CreateCcuBuffer();
        Executor executor = CreateExecutor();
        MaskSignal sig = CreateMaskSignal();

        LocalPost(sig, 1);
        RemoteWait(*transports[0], 0, 1);

        Memory loc = CreateMemory();
        Memory rmt = CreateMemory();
        Variable len = CreateVariable();
        Read(*transports[0], loc, rmt, len, sig, 1);
        ReadReduce(*transports[0], loc, rmt, len, DataType::FP32, ReduceOp::SUM, sig, 1);
        Write(*transports[0], rmt, loc, len, sig, 1);
        WriteReduce(*transports[0], rmt, loc, len, DataType::FP32, ReduceOp::SUM, sig, 1);

        Memory src = CreateMemory();
        Memory dst = CreateMemory();
        LocalCopy(dst, src, len, sig, 1);
        LocalReduce(dst, src, len, DataType::FP32, ReduceOp::SUM, sig, 1);
        GroupOpSize goSize = CreateGroupOpSize();
        Load(goSize);
        GroupCopy(dst, src, goSize);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        auto taskArg = dynamic_cast<const CcuTaskArgTest*>(&arg);
        auto goSize = CalGoSize(taskArg->size);
        return {goSize[0], goSize[1], goSize[2], goSize[3]};
    }
};

class CcuContextTestRepeat : public CcuContext {
public:
    CcuContextTestRepeat(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        Variable iter = CreateVariable();
        iter = 1;
        {
            Repeat rp(this, iter == 1);
            rp.Break();
        }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

class CcuContextTestFunction : public CcuContext {
public:
    CcuContextTestFunction(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        {
            FuncBlock fb(this, "test");
            Variable a = CreateVariable();
            std::vector<Variable> aV = {CreateVariable(), CreateVariable()};
            fb.DefineInArg(a);
            fb.DefineInArg(aV);
            fb.DefineOutArg(a);
            fb.DefineOutArg(aV);
        }
        Variable b = CreateVariable();
        std::vector<Variable> bV = {CreateVariable(), CreateVariable()};
        auto fc = Func("test");
        fc.SetInArg(b);
        fc.SetInArg(bV);
        fc.SetOutArg(b);
        fc.SetOutArg(bV);

        Variable funcAddr = CreateVariable();
        Func(funcAddr);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

HcclResult
CtxAllocCkeStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    ckeInfos.clear();
    ResInfo ckeInfo(0, num);
    ckeInfos.push_back(ckeInfo);
    return HcclResult::HCCL_SUCCESS;
}
HcclResult
CtxAllocXnStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    xnInfos.clear();
    ResInfo xnInfo(0, num);
    xnInfos.push_back(xnInfo);
    return HcclResult::HCCL_SUCCESS;
}

TEST_F(CcuContextTest, Test)
{
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    ;
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(CtxAllocCkeStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(CtxAllocXnStub));

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto c = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::shared_ptr<CcuTransport> t = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    t->AppendRes(3, 3);
    t->SetCntCke(cntCke);
    t->GetChannelId();
    t->GetDieId();
    t->GetLocCkeByIndex(0);
    t->GetLocCntCkeByIndex(0);
    t->GetLocXnByIndex(0);
}

TEST_F(CcuContextTest, CtxTest)
{
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::Destroy).stubs();
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    ;
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(CtxAllocCkeStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(CtxAllocXnStub));
    MOCKER(&CcuDeviceManager::GetLoopChannelId)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    std::vector<uint32_t> cntCke = {0, 1, 2};

    uint32_t rankId = 0;
    uint32_t rankSize = 8;
    CcuCtxArgTest ctxArg(rankId, rankSize);
    std::vector<std::shared_ptr<CcuTransport>> transportInstances;
    std::vector<CcuTransport*> transports;
    for (int i = 0; i < rankSize; i++) {
        if (i != rankId) {
            auto c = std::make_unique<CcuConnection>(
                linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
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

    std::vector<std::function<std::unique_ptr<CcuContext>(
        CcuCtxArg&, const std::vector<CcuTransport*>&, const CcuTransportGroup&)>>
        contextConstructors = {
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestTracker>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestVariable>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestMultiArgs>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestDataTransfer>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestFunction>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestRepeat>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextRS>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextAG>(arg, transports, transportGroup);
            },
        };
    for (int i = 0; i < contextConstructors.size(); i++) {
        auto ctx = contextConstructors[i](ctxArg, transports, transportGroup);
        ctx->Init();

        CcuResReq req = ctx->GetResourceRequest();

        ctx->DumpReprestation();
        ctx->GetInstrCount();
        std::array<uint16_t, 2> channels = {0, 0};
        std::pair<uint64_t, uint64_t> token_info = {0, 0};
        auto refManager = std::make_shared<CcuRepReferenceManager>(ctx->GetDieId());
        auto translator = CcuRepTranslator(0, ctx->GetDieId(), refManager, channels, token_info, 0);
        auto instrInfo = translator.Translate(ctx->GetRepSequence(), ctx->GetInstrId());
        ctx->SetCcuInstrInfo(instrInfo);

        CcuTaskArgTest taskArg(0, 0, 100);
        std::vector<CcuTaskParam> tmp;
        auto ret = ctx->GeneTaskParam(taskArg, tmp);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("GeneTaskParam is failed!");
        }
    }
}

class CcuCtxArgSharedRes : public CcuCtxArg {
public:
    explicit CcuCtxArgSharedRes(uint32_t id) : id(id) {}
    virtual ~CcuCtxArgSharedRes() = default;
    CcuCtxSignature GetCtxSignature() const override
    {
        CcuCtxSignature signature;
        signature.Append("Test");
        return signature;
    }
    uint32_t id{0};
};

class CcuTaskArgSharedRes : public CcuTaskArg {
public:
    explicit CcuTaskArgSharedRes() {}
};

class CcuContextTestSharesRes : public CcuContext {
public:
    CcuContextTestSharesRes(
        CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {
        id = ((CcuCtxArgSharedRes&)(arg)).id;
    }

protected:
    void Algorithm() override
    {
        if (id == 0) {
            Variable var = CreateVariable();
            Load(var);
            for (uint32_t otherId = 1; otherId < 3; otherId++) {
                Variable otherVar = ImportVariable("var" + std::to_string(otherId));
                MaskSignal otherSig = ImportMaskSignal("sig" + std::to_string(otherId));
                LocalCtxPost(otherSig, 1);
                LocalCtxPostVar(var, otherVar, otherSig, 1);
            }
        } else {
            Variable var = CreateVariable();
            ExportVariable(var, "var" + std::to_string(id));
            MaskSignal sig;
            ExportMaskSignal(sig, "sig" + std::to_string(id));
        }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg)
    {
        if (id == 0) {
            return {1024};
        } else {
            return {};
        }
    }

private:
    uint32_t id;
};

void CreateTaskShareRes(CcuContextTestSharesRes& ctx, std::pair<uint64_t, uint64_t> tokenInfo, uint64_t hbmTokenInfo)
{
    std::array<uint16_t, 2> channels = {0, 0};
    std::pair<uint64_t, uint64_t> token_info = {0, 0};
    auto refManager = std::make_shared<CcuRepReferenceManager>(ctx.GetDieId());
    auto translator = CcuRepTranslator(0, ctx.GetDieId(), refManager, channels, tokenInfo, hbmTokenInfo);
    auto instrInfo = translator.Translate(ctx.GetRepSequence(), ctx.GetInstrId());
    ctx.SetCcuInstrInfo(instrInfo);
    CcuTaskArgSharedRes taskArg;
    std::vector<CcuTaskParam> tmp;
    auto ret = ctx.GeneTaskParam(taskArg, tmp);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GeneTaskParam is failed!");
    }
}

TEST_F(CcuContextTest, TestSharedRes)
{
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::Destroy).stubs();
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    ;
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(CtxAllocCkeStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(CtxAllocXnStub));
    MOCKER(&CcuDeviceManager::GetLoopChannelId)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    std::vector<uint32_t> cntCke = {0, 1, 2};

    CcuTransport::CclBufferInfo locCclBufInfo;
    std::vector<CcuTransport*> transports0;
    auto c0
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    std::shared_ptr<CcuTransport> t0 = std::make_shared<CcuTransport>(nullptr, std::move(c0), locCclBufInfo);
    transports0.push_back(t0.get());

    std::vector<CcuTransport*> transports1;
    auto c1
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    std::shared_ptr<CcuTransport> t1 = std::make_shared<CcuTransport>(nullptr, std::move(c1), locCclBufInfo);
    transports1.push_back(t1.get());

    CcuTransportGroup transportGroup(transports0, 3);

    CcuCtxArgSharedRes ctxArg0(0);
    auto ctx0 = CcuContextTestSharesRes(ctxArg0, transports0, transportGroup);
    ctx0.Init();
    ctx0.DumpReprestation();
    ctx0.GetResourceRequest();
    auto res0 = ctx0.GetResource();
    for (int i = 0; i < res0.variable[ctx0.GetDieId()].size(); i++) {
        res0.variable[ctx0.GetDieId()][i].Reset(i);
    }

    CcuCtxArgSharedRes ctxArg1(1);
    auto ctx1 = CcuContextTestSharesRes(ctxArg1, transports0, transportGroup);
    ctx1.Init();
    ctx1.DumpReprestation();
    ctx1.GetResourceRequest();
    auto res1 = ctx1.GetResource();
    for (int i = 0; i < res1.maskSignal[ctx1.GetDieId()].size(); i++) {
        res1.maskSignal[ctx1.GetDieId()][i].Reset(256 + i);
    }
    for (int i = 0; i < res1.variable[ctx1.GetDieId()].size(); i++) {
        res1.variable[ctx1.GetDieId()][i].Reset(512 + i);
    }

    CcuCtxArgSharedRes ctxArg2(2);
    auto ctx2 = CcuContextTestSharesRes(ctxArg2, transports1, transportGroup);
    ctx2.Init();
    ctx2.DumpReprestation();
    ctx2.GetResourceRequest();
    auto res2 = ctx2.GetResource();
    for (int i = 0; i < res2.maskSignal[ctx2.GetDieId()].size(); i++) {
        res2.maskSignal[ctx2.GetDieId()][i].Reset(512 + i);
    }
    for (int i = 0; i < res2.variable[ctx2.GetDieId()].size(); i++) {
        res2.variable[ctx2.GetDieId()][i].Reset(1024 + i);
    }

    auto importRes0 = ctx0.GetImportRes();

    auto exportRes1 = ctx1.GetExportRes();
    auto exportRes2 = ctx2.GetExportRes();

    for (auto& r : importRes0.sharedVar) {
        HCCL_INFO("importRes0.sharedVar %s", r.first.c_str());
        auto it = exportRes1.sharedVar.find(r.first);
        if (it != exportRes1.sharedVar.end()) {
            r.second.Reset(it->second.Id(), it->second.DieId());
        } else {
            auto it1 = exportRes2.sharedVar.find(r.first);
            if (it1 != exportRes2.sharedVar.end()) {
                r.second.Reset(it1->second.Id(), it1->second.DieId());
            }
        }
    }
    for (auto& r : importRes0.sharedSig) {
        HCCL_INFO("importRes0.sharedSig %s", r.first.c_str());
        auto it = exportRes1.sharedSig.find(r.first);
        if (it != exportRes1.sharedSig.end()) {
            r.second.Reset(it->second.Id(), it->second.DieId());
        } else {
            auto it1 = exportRes2.sharedSig.find(r.first);
            if (it1 != exportRes2.sharedSig.end()) {
                r.second.Reset(it1->second.Id(), it1->second.DieId());
            }
        }
    }

    CreateTaskShareRes(ctx0, {0, 0}, 0);
    CreateTaskShareRes(ctx1, {1, 1}, 1);
    CreateTaskShareRes(ctx2, {1, 1}, 2);
}
