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

#include "ccu_rep.h"
#include "ccu_ctx.h"
#include "ccu_rep_translator.h"

#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#undef protected
#undef private

#include "log.h"
using namespace Hccl;
using namespace CcuRep;

class CcuContextTrackerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuContextTrackerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuContextTrackerTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        JfcHandle jfcHandle = 1;
        MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
        std::cout << "A Test case in CcuContextTrackerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuContextTrackerTest TearDown" << std::endl;
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

class CcuContextTrackerTestTracker : public CcuContext {
public:
    CcuContextTrackerTestTracker(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        var = CreateVariable();
        addr = CreateAddress();
        mem = CreateMemory();
        goSize = CreateGroupOpSize();
        auto var1 = CreateVariable();
        var = var1;
        auto addr1 = CreateAddress();
        addr = addr1;
        goSize.addrOffset = var1;
        {
            auto t = CreateVariable();
        }
        {
            auto t = CreateAddress();
        }
        {
            auto t = CreateMemory();
        }
        {
            auto token = CreateVariable();
            auto t = CreateMemory(token);
        }
        {
            auto t = CreateMaskSignal();
        }
        {
            auto t = CreateCcuBuffer();
        }
        {
            auto t = CreateExecutor();
        }
        {
            auto t = CreateBlockCcuBuffer(12);
        }
        {
            auto t = CreateBlockExecutor(24);
        }
        {
            auto t = CreateBlockMaskSignal(36);
        }
        {
            auto t = CreateGroupOpSize();
        }
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }

private:
    Variable var;
    Address addr;
    Memory mem;
    GroupOpSize goSize;
};

HcclResult
CtxAllocCkeStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos);
HcclResult
CtxAllocXnStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos);

TEST_F(CcuContextTrackerTest, TrackerTest)
{
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::Destroy).stubs();
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(CtxAllocCkeStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(CtxAllocXnStub));
    MOCKER(&CcuDeviceManager::GetLoopChannelId)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::GetXnBaseAddr).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));

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
                return std::make_unique<CcuContextTrackerTestTracker>(arg, transports, transportGroup);
            },
        };
    std::array<uint16_t, 2> channels = {0, 0};
    std::pair<uint64_t, uint64_t> token_info = {0, 0};
    for (int i = 0; i < contextConstructors.size(); i++) {
        auto ctx = contextConstructors[i](ctxArg, transports, transportGroup);
        ctx->Init();

        CcuResReq req = ctx->GetResourceRequest();

        ctx->DumpReprestation();
        ctx->GetInstrCount();

        auto refManager = std::make_shared<CcuRepReferenceManager>(0);
        auto translator = CcuRepTranslator(0, 0, refManager, channels, token_info, 0);
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
