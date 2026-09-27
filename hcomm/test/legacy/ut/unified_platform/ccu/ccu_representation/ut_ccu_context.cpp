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
#include "ccu_context_common.h"
#undef protected
#undef private

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
                return std::make_unique<CcuContextTestVariable>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestMultiArgs>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestDataTransfer>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestCondition>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestRepeat>(arg, transports, transportGroup);
            },
            [](CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup) {
                return std::make_unique<CcuContextTestFunction>(arg, transports, transportGroup);
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
        CHK_PRT_RET_NULL(ctx->Init(), HCCL_INFO("Init is fail"));

        CcuResReq req = ctx->GetResourceRequest();

        ctx->DumpReprestation();
        ctx->GetInstrCount();

        auto refManager = std::make_shared<CcuRepReferenceManager>(ctx->GetDieId());
        std::array<uint16_t, 2> channels = {0, 0};
        std::pair<uint64_t, uint64_t> token_info = {0, 0};

        auto translator = CcuRepTranslator(0, ctx->GetDieId(), refManager, channels, token_info, 0);
        translator.var[0].Reset(3071);
        translator.addr[0].Reset(3071);
        translator.signal[0].Reset(1023);
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

TEST_F(CcuContextTest, TestSharedRes)
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
    auto c = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    std::vector<uint32_t> cntCke = {0, 1, 2};
    CcuTransport::CclBufferInfo locCclBufInfo;
    std::vector<CcuTransport*> transports0;
    std::shared_ptr<CcuTransport> t0 = std::make_shared<CcuTransport>(nullptr, std::move(c), locCclBufInfo);
    transports0.push_back(t0.get());

    auto c0
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

    std::vector<CcuTransport*> transports1;
    std::shared_ptr<CcuTransport> t1 = std::make_shared<CcuTransport>(nullptr, std::move(c0), locCclBufInfo);
    transports1.push_back(t1.get());

    CcuTransportGroup transportGroup(transports0, 3);

    CcuCtxArgSharedRes ctxArg0(0);
    auto ctx0 = CcuContextTestSharesRes(ctxArg0, transports0, transportGroup);
    CHK_PRT_RET_NULL(ctx0.Init(), HCCL_INFO("Init is fail"));
    ctx0.DumpReprestation();
    ctx0.GetResourceRequest();
    auto res0 = ctx0.GetResource();
    for (int i = 0; i < res0.variable[ctx0.GetDieId()].size(); i++) {
        res0.variable[ctx0.GetDieId()][i].Reset(i);
    }

    CcuCtxArgSharedRes ctxArg1(1);
    auto ctx1 = CcuContextTestSharesRes(ctxArg1, transports0, transportGroup);
    CHK_PRT_RET_NULL(ctx1.Init(), HCCL_INFO("Init is fail"));
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
    CHK_PRT_RET_NULL(ctx2.Init(), HCCL_INFO("Init is fail"));
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

    auto refManager0 = std::make_shared<CcuRepReferenceManager>(ctx0.GetDieId());
    std::array<uint16_t, 2> channels0 = {0, 0};
    std::pair<uint64_t, uint64_t> token_info = {0, 0};
    auto translator0 = CcuRepTranslator(0, ctx0.GetDieId(), refManager0, channels0, token_info, 0);
    auto instrInfo0 = translator0.Translate(ctx0.GetRepSequence(), ctx0.GetInstrId());
    ctx0.SetCcuInstrInfo(instrInfo0);
    CcuTaskArgSharedRes taskArg0;
    std::vector<CcuTaskParam> tmp0;
    auto ret0 = ctx0.GeneTaskParam(taskArg0, tmp0);
    if (ret0 != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GeneTaskParam is failed!");
    }

    auto refManager1 = std::make_shared<CcuRepReferenceManager>(ctx1.GetDieId());
    std::array<uint16_t, 2> channels1 = {0, 0};
    std::pair<uint64_t, uint64_t> token_info1 = {0, 0};
    auto translator1 = CcuRepTranslator(0, ctx1.GetDieId(), refManager1, channels1, token_info1, 1);
    auto instrInfo1 = translator1.Translate(ctx1.GetRepSequence(), ctx1.GetInstrId());
    ctx1.SetCcuInstrInfo(instrInfo1);
    CcuTaskArgSharedRes taskArg1;
    std::vector<CcuTaskParam> tmp1;
    auto ret1 = ctx1.GeneTaskParam(taskArg1, tmp1);
    if (ret1 != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GeneTaskParam is failed!");
    }

    auto refManager2 = std::make_shared<CcuRepReferenceManager>(ctx2.GetDieId());
    std::array<uint16_t, 2> channels2 = {0, 0};
    std::pair<uint64_t, uint64_t> token_info2 = {0, 0};
    auto translator2 = CcuRepTranslator(0, ctx2.GetDieId(), refManager2, channels2, token_info2, 2);
    auto instrInfo2 = translator2.Translate(ctx2.GetRepSequence(), ctx2.GetInstrId());
    ctx2.SetCcuInstrInfo(instrInfo2);
    CcuTaskArgSharedRes taskArg2;
    std::vector<CcuTaskParam> tmp2;
    auto ret2 = ctx2.GeneTaskParam(taskArg2, tmp2);
    if (ret2 != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GeneTaskParam is failed!");
    }
}

class CcuContextTestLocalReduce : public CcuContext {
public:
    CcuContextTestLocalReduce(DataType inputDataType, DataType outputDataType, ReduceOp opType)
        : inputDataType(inputDataType),
          outputDataType(outputDataType),
          opType(opType)
    {}

protected:
    void Algorithm() override
    {
        std::vector<CcuRep::CcuBuffer> bufs(8);
        MaskSignal sig;
        Variable len;
        LocalReduce(bufs, 8, inputDataType, outputDataType, opType, sig, len, 1);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }

private:
    DataType inputDataType;
    DataType outputDataType;
    ReduceOp opType;
};

TEST_F(CcuContextTest, LocalReduce)
{
    EXPECT_EQ(CcuContextTestLocalReduce(DataType::FP32, DataType::FP32, ReduceOp::SUM).Init(), HCCL_SUCCESS);
    EXPECT_EQ(CcuContextTestLocalReduce(DataType::FP16, DataType::FP16, ReduceOp::SUM).Init(), HCCL_SUCCESS);

    EXPECT_EQ(CcuContextTestLocalReduce(DataType::INT8, DataType::FP16, ReduceOp::SUM).Init(), HCCL_SUCCESS);

    EXPECT_EQ(CcuContextTestLocalReduce(DataType::INT8, DataType::INT8, ReduceOp::MAX).Init(), HCCL_SUCCESS);

    EXPECT_EQ(CcuContextTestLocalReduce(DataType::INT8, DataType::INT8, ReduceOp::SUM).Init(), HCCL_E_INTERNAL);
    EXPECT_EQ(CcuContextTestLocalReduce(DataType::FP16, DataType::FP32, ReduceOp::SUM).Init(), HCCL_E_INTERNAL);
    EXPECT_EQ(CcuContextTestLocalReduce(DataType::FP16, DataType::FP32, ReduceOp::MAX).Init(), HCCL_E_INTERNAL);
    EXPECT_EQ(CcuContextTestLocalReduce(DataType::INT64, DataType::INT64, ReduceOp::MAX).Init(), HCCL_E_INTERNAL);
}
