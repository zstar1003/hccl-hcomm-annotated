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
#include "ccu_ctx_mgr.h"
#include "ins_exe_que.h"

#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"

#include "log.h"
using namespace Hccl;
using namespace CcuRep;

class CcuContextManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuContextManagerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuContextManagerTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        JfcHandle jfcHandle = 1;
        MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));
        std::cout << "A Test case in CcuContextManagerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuContextManagerTest TearDown" << std::endl;
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
        std::vector<Variable> input(size);
        std::vector<Variable> output(size);
        std::vector<Variable> token(size);

        Variable offset;
        GroupOpSize goSize;

        Memory src;
        std::vector<Memory> dst(size);

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
        std::vector<Variable> input(size);
        std::vector<Variable> output(size);
        std::vector<Variable> token(size);

        Variable offset;
        GroupOpSize goSize;

        std::vector<Memory> src(size);
        Memory dst;

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
        for (int i = 0; i < 14; i++) {
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

class CcuContextTestVariable : public CcuContext {
public:
    CcuContextTestVariable(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
        : CcuContext(arg, transports, transportGroup)
    {}

protected:
    void Algorithm() override
    {
        Variable a;
        Variable b;
        Variable c;
        a = b + c;

        Address aa;
        Address ab;
        Address ac;
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
        CcuRep::CcuBuffer buf;
        Executor executor;
        MaskSignal sig;

        LocalPost(sig, 1);
        RemoteWait(*transports[0], 0, 1);

        Memory loc;
        Memory rmt;
        Variable len;
        Read(*transports[0], loc, rmt, len, sig, 1);
        ReadReduce(*transports[0], loc, rmt, len, DataType::FP32, ReduceOp::SUM, sig, 1);
        Write(*transports[0], rmt, loc, len, sig, 1);
        WriteReduce(*transports[0], rmt, loc, len, DataType::FP32, ReduceOp::SUM, sig, 1);

        Memory src;
        Memory dst;
        LocalCopy(dst, src, len, sig, 1);
        LocalReduce(dst, src, len, DataType::FP32, ReduceOp::SUM, sig, 1);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
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
        Variable iter;
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
            Variable a;
            std::vector<Variable> aV(2);
            fb.DefineInArg(a);
            fb.DefineInArg(aV);
            fb.DefineOutArg(a);
            fb.DefineOutArg(aV);
        }
        Variable b;
        std::vector<Variable> bV(2);
        auto fc = Func("test");
        fc.SetInArg(b);
        fc.SetInArg(bV);
        fc.SetOutArg(b);
        fc.SetOutArg(bV);

        Variable funcAddr;
        Func(funcAddr);
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) { return {}; }
};

HcclResult
CtxMgrAllocCkeStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    ckeInfos.clear();
    ResInfo ckeInfo(0, num);
    ckeInfos.push_back(ckeInfo);
    return HcclResult::HCCL_SUCCESS;
}
HcclResult
CtxMgrAllocXnStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    xnInfos.clear();
    ResInfo xnInfo(0, num);
    xnInfos.push_back(xnInfo);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrAllocResHandleStub(const int32_t deviceLogicId, const CcuResReq resReq, CcuResHandle& handle)
{
    handle = reinterpret_cast<CcuResHandle>(0x100);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrGetResourceStub(const int32_t deviceLogicId, const CcuResHandle handle, CcuResRepository& ccuResRepo)
{
    ccuResRepo.blockMs[0].resize(1);
    ccuResRepo.blockMs[0][0].startId = 0;
    ccuResRepo.blockMs[0][0].num = 1024;

    ccuResRepo.cke[0].resize(1);
    ccuResRepo.cke[0][0].startId = 0;
    ccuResRepo.cke[0][0].num = 32;

    ccuResRepo.blockCke[0].resize(1);
    ccuResRepo.blockCke[0][0].startId = 0;
    ccuResRepo.blockCke[0][0].num = 128;

    ccuResRepo.blockLoopEngine[0].resize(1);
    ccuResRepo.blockLoopEngine[0][0].startId = 0;
    ccuResRepo.blockLoopEngine[0][0].num = 64;

    ccuResRepo.gsa[0].resize(1);
    ccuResRepo.gsa[0][0].startId = 0;
    ccuResRepo.gsa[0][0].num = 48;

    ccuResRepo.xn[0].resize(1);
    ccuResRepo.xn[0][0].startId = 0;
    ccuResRepo.xn[0][0].num = 1248;

    ccuResRepo.mission.mission[0].resize(1);
    ccuResRepo.mission.mission[0][0].startId = 0;
    ccuResRepo.mission.mission[0][0].num = 1;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrReleaseResHandleStub(const int32_t deviceLogicId, const CcuResHandle handle)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrAllocInsStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    insInfo = ResInfo(0, num);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrReleaseInsStub(const int32_t deviceLogicId, const uint8_t dieId, ResInfo& insInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrGetMissionKeyStub(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& missionKey)
{
    missionKey = 0xFF;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrGetInstructionNumStub(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& instrNum)
{
    instrNum = 0xFF;
    return HcclResult::HCCL_SUCCESS;
}

TEST_F(CcuContextManagerTest, AGTest)
{
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::Destroy).stubs();
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    ;
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(CtxMgrAllocCkeStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(CtxMgrAllocXnStub));
    MOCKER_CPP(&CcuDeviceManager::GetCcuResourceSpaceTokenInfoForLocal)
        .stubs()
        .will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuDeviceManager::GetCcuResourceSpaceTokenInfo)
        .stubs()
        .will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocResHandle).stubs().will(invoke(CtxMgrAllocResHandleStub));
    MOCKER(CcuDeviceManager::GetResource).stubs().will(invoke(CtxMgrGetResourceStub));
    MOCKER(CcuDeviceManager::ReleaseResHandle).stubs().will(invoke(CtxMgrReleaseResHandleStub));
    MOCKER(CcuDeviceManager::AllocIns).stubs().will(invoke(CtxMgrAllocInsStub));
    MOCKER(CcuDeviceManager::ReleaseIns).stubs().will(invoke(CtxMgrReleaseInsStub));
    MOCKER(CcuDeviceManager::GetInstructionNum).stubs().will(invoke(CtxMgrGetInstructionNumStub));
    MOCKER(CcuDeviceManager::GetMissionKey).stubs().will(invoke(CtxMgrGetMissionKeyStub));

    MOCKER(&CcuDeviceManager::GetLoopChannelId)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(&CcuDeviceManager::GetXnBaseAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(&CcuDeviceManager::GetCcuResourceSpaceTokenInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(HrtMemcpy).stubs();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

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

    CcuContextAG ctx(ctxArg, transports, transportGroup);

    // 申请资源
    s32 deviceLogicId = 0;
    CcuCtxGroup ctxGroup;
    ctxGroup.ctxs.push_back(std::make_unique<CcuContextAG>(ctx));

    CcuResPack resPack;

    EXPECT_EQ(CcuCtxMgr::AllocRes(deviceLogicId, ctxGroup, resPack), HCCL_SUCCESS);

    // 注册指令
    InsExeQue::ExtInsExeEntity entity;
    entity.ctxGroup = std::move(ctxGroup);
    InsExeQue::ExtInsExeEntityId entityId = 0;

    EXPECT_EQ(InsExeQue::RegisterExtendInstruction(deviceLogicId, entity, entityId), HCCL_SUCCESS);

    // 获取taskPara参数
    CcuTaskArgTest taskArg(0, 0, 100);
    std::vector<std::vector<CcuTaskParam>> taskParam;

    EXPECT_EQ(CcuCtxMgr::GetTaskParam(deviceLogicId, taskArg, entityId, taskParam), HCCL_SUCCESS);
    EXPECT_EQ(taskParam.size(), 1);

    // 释放资源
    // CcuCtxMgr::ReleaseRes(deviceLogicId, entity.ctxGroup);

    // 卸载指令
    EXPECT_EQ(InsExeQue::DeregisterExtendInstruction(deviceLogicId, entityId), HCCL_SUCCESS);
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
            Variable var;
            Load(var);
            for (uint32_t otherId = 1; otherId < 3; otherId++) {
                Variable otherVar = ImportVariable("var" + std::to_string(otherId));
                MaskSignal otherSig = ImportMaskSignal("sig" + std::to_string(otherId));
                LocalCtxPost(otherSig, 1);
                LocalCtxPostVar(var, otherVar, otherSig, 1);
            }
        } else {
            Variable var;
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

HcclResult
CtxMgrGetResourceSharedResStub(const int32_t deviceLogicId, const CcuResHandle handle, CcuResRepository& ccuResRepo)
{
    ccuResRepo.cke[0].resize(1);
    ccuResRepo.cke[0][0].startId = 0;
    ccuResRepo.cke[0][0].num = 3;

    ccuResRepo.xn[0].resize(1);
    ccuResRepo.xn[0][0].startId = 0;
    ccuResRepo.xn[0][0].num = 45;

    ccuResRepo.gsa[0].resize(1);
    ccuResRepo.gsa[0][0].startId = 0;
    ccuResRepo.gsa[0][0].num = 1;

    ccuResRepo.mission.mission[0].resize(1);
    ccuResRepo.mission.mission[0][0].startId = 0;
    ccuResRepo.mission.mission[0][0].num = 3;

    return HcclResult::HCCL_SUCCESS;
}

TEST_F(CcuContextManagerTest, TestSharedRes)
{
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::Destroy).stubs();
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    ;
    MOCKER_CPP(&CcuDeviceManager::GetCcuResourceSpaceTokenInfoForLocal)
        .stubs()
        .will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuDeviceManager::GetCcuResourceSpaceTokenInfo)
        .stubs()
        .will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(CtxMgrAllocCkeStub));
    MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(CtxMgrAllocXnStub));

    MOCKER(CcuDeviceManager::AllocResHandle).stubs().will(invoke(CtxMgrAllocResHandleStub));
    MOCKER(CcuDeviceManager::GetResource).stubs().will(invoke(CtxMgrGetResourceSharedResStub));
    MOCKER(CcuDeviceManager::ReleaseResHandle).stubs().will(invoke(CtxMgrReleaseResHandleStub));
    MOCKER(CcuDeviceManager::AllocIns).stubs().will(invoke(CtxMgrAllocInsStub));
    MOCKER(CcuDeviceManager::ReleaseIns).stubs().will(invoke(CtxMgrReleaseInsStub));
    MOCKER(CcuDeviceManager::GetInstructionNum).stubs().will(invoke(CtxMgrGetInstructionNumStub));
    MOCKER(CcuDeviceManager::GetMissionKey).stubs().will(invoke(CtxMgrGetMissionKeyStub));

    MOCKER(&CcuDeviceManager::GetLoopChannelId)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(HrtMemcpy).stubs();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

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

    CcuCtxArgSharedRes ctxArg1(1);
    auto ctx1 = CcuContextTestSharesRes(ctxArg1, transports0, transportGroup);

    CcuCtxArgSharedRes ctxArg2(2);
    auto ctx2 = CcuContextTestSharesRes(ctxArg2, transports1, transportGroup);

    // 申请资源
    s32 deviceLogicId = 0;
    CcuCtxGroup ctxGroup;
    ctxGroup.ctxs.push_back(std::make_unique<CcuContextTestSharesRes>(ctx0));
    ctxGroup.ctxs.push_back(std::make_unique<CcuContextTestSharesRes>(ctx1));
    ctxGroup.ctxs.push_back(std::make_unique<CcuContextTestSharesRes>(ctx2));

    CcuResPack resPack;

    EXPECT_EQ(CcuCtxMgr::AllocRes(deviceLogicId, ctxGroup, resPack), HCCL_SUCCESS);

    // 注册指令
    InsExeQue::ExtInsExeEntity entity;
    entity.ctxGroup = std::move(ctxGroup);
    InsExeQue::ExtInsExeEntityId entityId = 0;

    EXPECT_EQ(InsExeQue::RegisterExtendInstruction(deviceLogicId, entity, entityId), HCCL_SUCCESS);

    // 获取taskPara参数
    CcuTaskArgTest taskArg(0, 0, 100);
    std::vector<std::vector<CcuTaskParam>> taskParam;

    EXPECT_EQ(CcuCtxMgr::GetTaskParam(deviceLogicId, taskArg, entityId, taskParam), HCCL_SUCCESS);
    EXPECT_EQ(taskParam.size(), 3);

    // 释放资源
    // CcuCtxMgr::ReleaseRes(deviceLogicId, entity.ctxGroup);

    // 卸载指令
    EXPECT_EQ(InsExeQue::DeregisterExtendInstruction(deviceLogicId, entityId), HCCL_SUCCESS);
}
