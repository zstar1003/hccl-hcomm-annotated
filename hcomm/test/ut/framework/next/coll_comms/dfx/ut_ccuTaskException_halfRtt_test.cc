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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hccl_comm_pub.h"
#include "hccl_api_base_test.h"
#include "ccu_comp.h"
#define private public
#include "hcclCommTaskException.h"
#include "ccuTaskException.h"
#undef private

#include "ccu_ins_generator_v2.h"
#include "ccu_rep_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_rep_context_v1.h"
#include "ccu_res_repo.h"

using namespace hccl;
using namespace hcomm;
using namespace hcomm::CcuRep;

// GetCcuXnValue 查询失败时的返回值(与 res_pub.h 中 DFX_INVALID_U64 一致)
constexpr uint64_t INVALID_U64_VAL = UINT64_MAX;

class CcuTaskExceptionHalfRttTest : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

protected:
    // 统一 mock 硬件查询通道: GetCcuXnValue 内部调用 HccpRaTlvRequestForCustomChannel
    void MockXnQuerySuccess()
    {
        MOCKER(HccpRaTlvRequestForCustomChannel)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
    }

    void MockXnQueryFail()
    {
        MOCKER(HccpRaTlvRequestForCustomChannel)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_E_PARA));
    }

    Hccl::TaskInfo MakeTaskInfo()
    {
        Hccl::ParaCcu paraCcu = {};
        Hccl::TaskParam taskParam
            = {.taskType = Hccl::TaskParamType::TASK_CCU,
               .beginTime = 0,
               .endTime = 0,
               .isMaster = false,
               .taskPara = {.Ccu = paraCcu},
               .ccuDetailInfo = nullptr};
        return Hccl::TaskInfo(1, 2, 3, taskParam, nullptr, false);
    }

    ErrorInfoBase MakeBaseInfo() { return ErrorInfoBase{0, 0, 0, 0x10, 0x01}; }

    CcuInsGeneratorV2 insGen{};
    CcuRepContext context{};
};

/* ---------------- GetCcuErrorMsgByType: 6 个新类型分发 ---------------- */

TEST_F(CcuTaskExceptionHalfRttTest, GetCcuErrorMsgByType_WriteVarAtomic)
{
    CcuErrorInfo info{};
    info.type = CcuErrorType::WRITE_VAR_ATOMIC;
    info.SetBaseInfo(CcuRepType::WRITE_VAR_ATOMIC, 0, 0, 42);
    info.msg.writeVarAtomic.dstId = 1;
    info.msg.writeVarAtomic.dstValue = 0x100;
    info.msg.writeVarAtomic.signalId = 2;
    info.msg.writeVarAtomic.signalMask = 0xF;

    Hccl::TaskInfo taskInfo = MakeTaskInfo();
    std::string msg = CcuTaskException::GetCcuErrorMsgByType(info, taskInfo, 0);
    EXPECT_TRUE(msg.find("WriteVarAtomic") != std::string::npos);
    EXPECT_TRUE(msg.find("dstId[1]") != std::string::npos);
    EXPECT_TRUE(msg.find("dstValue[0x0000000000000100]") != std::string::npos);
    EXPECT_TRUE(msg.find("sem[2]") != std::string::npos);
    EXPECT_TRUE(msg.find("mask[0x000f]") != std::string::npos);
}

TEST_F(CcuTaskExceptionHalfRttTest, GetCcuErrorMsgByType_WriteWithCntInc)
{
    CcuErrorInfo info{};
    info.type = CcuErrorType::WRITE_WITH_CNT_INC;
    info.SetBaseInfo(CcuRepType::WRITE_WITH_CNT_INC, 0, 0, 43);
    info.msg.writeWithCntInc.locAddr = 0x1000;
    info.msg.writeWithCntInc.rmtAddr = 0x2000;
    info.msg.writeWithCntInc.len = 4096;
    info.msg.writeWithCntInc.incCntAddr = 0x3000;

    Hccl::TaskInfo taskInfo = MakeTaskInfo();
    std::string msg = CcuTaskException::GetCcuErrorMsgByType(info, taskInfo, 0);
    EXPECT_TRUE(msg.find("WriteWithCntInc") != std::string::npos);
    EXPECT_TRUE(msg.find("Memory[0x0000000000001000]") != std::string::npos);
    EXPECT_TRUE(msg.find("Len[4096]") != std::string::npos);
    EXPECT_TRUE(msg.find("incCntAddr[0x0000000000003000]") != std::string::npos);
}

TEST_F(CcuTaskExceptionHalfRttTest, GetCcuErrorMsgByType_CascCntWait)
{
    CcuErrorInfo info{};
    info.type = CcuErrorType::CASC_CNT_WAIT;
    info.SetBaseInfo(CcuRepType::CASC_CNT_WAIT, 0, 0, 44);
    info.msg.cascCntWait.targetValue = 8;
    info.msg.cascCntWait.currentCntValue = 5;

    Hccl::TaskInfo taskInfo = MakeTaskInfo();
    std::string msg = CcuTaskException::GetCcuErrorMsgByType(info, taskInfo, 0);
    EXPECT_TRUE(msg.find("CascCntWait") != std::string::npos);
    EXPECT_TRUE(msg.find("targetValue[0x0000000000000008]") != std::string::npos);
    EXPECT_TRUE(msg.find("currentCntValue[0x0000000000000005]") != std::string::npos);
}

TEST_F(CcuTaskExceptionHalfRttTest, GetCcuErrorMsgByType_CascCntClear)
{
    CcuErrorInfo info{};
    info.type = CcuErrorType::CASC_CNT_CLEAR;
    info.SetBaseInfo(CcuRepType::CASC_CNT_CLEAR, 0, 0, 45);
    info.msg.cascCntClear.wishCntXnIdFirst = 100;
    info.msg.cascCntClear.expectedCntXn = 200;

    Hccl::TaskInfo taskInfo = MakeTaskInfo();
    std::string msg = CcuTaskException::GetCcuErrorMsgByType(info, taskInfo, 0);
    EXPECT_TRUE(msg.find("CascCntClear") != std::string::npos);
    EXPECT_TRUE(msg.find("Xns[100~200]") != std::string::npos);
}

TEST_F(CcuTaskExceptionHalfRttTest, GetCcuErrorMsgByType_LoadAddImm)
{
    CcuErrorInfo info{};
    info.type = CcuErrorType::LOAD_ADD_IMM;
    info.SetBaseInfo(CcuRepType::LOAD_ADD_IMM, 0, 0, 46);
    info.msg.loadAddImm.srcId = 1;
    info.msg.loadAddImm.srcOffsetId = 2;
    info.msg.loadAddImm.dstId = 3;
    info.msg.loadAddImm.immAddValue = 0x55;
    info.msg.loadAddImm.srcValue = 0x100;
    info.msg.loadAddImm.srcOffsetValue = 0x200;

    Hccl::TaskInfo taskInfo = MakeTaskInfo();
    std::string msg = CcuTaskException::GetCcuErrorMsgByType(info, taskInfo, 0);
    EXPECT_TRUE(msg.find("LoadAddImm") != std::string::npos);
    EXPECT_TRUE(msg.find("srcId[1]") != std::string::npos);
    EXPECT_TRUE(msg.find("immAddValue[85]") != std::string::npos);
    EXPECT_TRUE(msg.find("srcValue[0x0000000000000100]") != std::string::npos);
}

TEST_F(CcuTaskExceptionHalfRttTest, GetCcuErrorMsgByType_StoreAddImm)
{
    CcuErrorInfo info{};
    info.type = CcuErrorType::STORE_ADD_IMM;
    info.SetBaseInfo(CcuRepType::STORE_ADD_IMM, 0, 0, 47);
    info.msg.storeAddImm.dstId = 4;
    info.msg.storeAddImm.dstOffsetId = 5;
    info.msg.storeAddImm.srcId = 6;
    info.msg.storeAddImm.immAddValue = 0x66;
    info.msg.storeAddImm.dstValue = 0x300;
    info.msg.storeAddImm.dstOffsetValue = 0x400;

    Hccl::TaskInfo taskInfo = MakeTaskInfo();
    std::string msg = CcuTaskException::GetCcuErrorMsgByType(info, taskInfo, 0);
    EXPECT_TRUE(msg.find("StoreAddImm") != std::string::npos);
    EXPECT_TRUE(msg.find("dstId[4]") != std::string::npos);
    EXPECT_TRUE(msg.find("dstValue[0x0000000000000300]") != std::string::npos);
}

/* ---------------- GenErrorInfo: 从 rep 提取字段(含硬件查询 mock) ---------------- */

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoWriteVarAtomic_ExtractFields)
{
    MockXnQuerySuccess();

    Variable channel{&context};
    Address addr;
    Variable token;
    RemoteAddr varAddr{addr, token};
    Variable targetValue{&context};
    CompletedEvent sem{&context};
    auto rep = std::make_shared<CcuRepWriteVarAtomic>(&insGen, channel, varAddr, targetValue, sem, 0xF);

    std::vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenErrorInfoWriteVarAtomic(MakeBaseInfo(), rep, errorInfo);

    ASSERT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WRITE_VAR_ATOMIC);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::WRITE_VAR_ATOMIC);
    EXPECT_EQ(errorInfo[0].msg.writeVarAtomic.dstId, targetValue.Id());
    EXPECT_EQ(errorInfo[0].msg.writeVarAtomic.signalId, sem.Id());
    EXPECT_EQ(errorInfo[0].msg.writeVarAtomic.signalMask, 0xF);
    // mock 查询成功时, Xn 寄存器值读出为 0
    EXPECT_EQ(errorInfo[0].msg.writeVarAtomic.dstValue, 0u);
}

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoWriteVarAtomic_QueryFailReturnsInvalid)
{
    MockXnQueryFail();

    Address addr;
    Variable token;
    Variable channel{&context};
    Variable targetValue{&context};
    CompletedEvent sem{&context};
    RemoteAddr varAddr{addr, token};
    auto rep = std::make_shared<CcuRepWriteVarAtomic>(&insGen, channel, varAddr, targetValue, sem, 0xF);

    std::vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenErrorInfoWriteVarAtomic(MakeBaseInfo(), rep, errorInfo);

    ASSERT_EQ(errorInfo.size(), 1u);
    // 查询失败时返回 INVALID_U64(容错不崩溃)
    EXPECT_EQ(errorInfo[0].msg.writeVarAtomic.dstValue, INVALID_U64_VAL);
}

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoWriteWithCntInc_ExtractFields)
{
    MockXnQuerySuccess();

    Variable channel{&context};
    Address addr;
    Variable token;
    RemoteAddr rem{addr, token};
    LocalAddr loc{addr, token};
    Variable len{&context};
    RemoteAddr incCnt{addr, token};
    auto rep = std::make_shared<CcuRepWriteWithCntInc>(&insGen, channel, rem, loc, len, incCnt);

    std::vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenErrorInfoWriteWithCntInc(MakeBaseInfo(), rep, errorInfo);

    ASSERT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WRITE_WITH_CNT_INC);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::WRITE_WITH_CNT_INC);
    // mock 查询成功时所有 Xn 值为 0, 验证查询链路被正确调用(字段被赋值)
    EXPECT_EQ(errorInfo[0].msg.writeWithCntInc.locAddr, 0u);
    EXPECT_EQ(errorInfo[0].msg.writeWithCntInc.rmtAddr, 0u);
    EXPECT_EQ(errorInfo[0].msg.writeWithCntInc.len, 0u);
    EXPECT_EQ(errorInfo[0].msg.writeWithCntInc.incCntAddr, 0u);
}

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoCascCntWait_ExtractTargetValue)
{
    MockXnQuerySuccess();
    // 构造测试块数据, totalCntXn=1022, 通过 rep 公有接口 SetCascCntBlock 按值注入
    CntXnBlock block{};
    block.wishCntXns = {0, 1021};
    block.totalCntXn = 1022;
    block.expectedCntXn = 1023;

    CcuRepCascCntWait rep(&insGen, 8888, 100);
    rep.SetCascCntBlock(block);
    auto repBase = std::make_shared<CcuRepCascCntWait>(std::move(rep));

    std::vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenErrorInfoCascCntWait(MakeBaseInfo(), repBase, errorInfo);

    ASSERT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::CASC_CNT_WAIT);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::CASC_CNT_WAIT);
    // 目标值来自 rep(编译期常量), 不依赖硬件查询
    EXPECT_EQ(errorInfo[0].msg.cascCntWait.targetValue, 100u);
    // 注入 totalCntXn=1022, mock 查询成功后 currentCntValue 被赋值(读出为 0)
    EXPECT_EQ(errorInfo[0].msg.cascCntWait.currentCntValue, 0u);
}

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoCascCntClear_ExtractWishCntXnId)
{
    // 构造测试块数据: wishCntXns = {1022, 1023}, 通过 rep 公有接口按值注入
    CntXnBlock block{};
    block.wishCntXns = {1022, 1023};
    block.expectedCntXn = 1025;

    CompletedEvent sem{&context};
    auto rep = std::make_shared<CcuRepCascCntClear>(&insGen, 9999, sem, 0xF);
    rep->SetCascCntBlock(block);

    std::vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenErrorInfoCascCntClear(MakeBaseInfo(), rep, errorInfo);

    ASSERT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::CASC_CNT_CLEAR);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::CASC_CNT_CLEAR);
    // 注入数据 wishCntXns = {1022, 1023}
    EXPECT_EQ(errorInfo[0].msg.cascCntClear.wishCntXnIdFirst, 1022u);
    EXPECT_EQ(errorInfo[0].msg.cascCntClear.expectedCntXn, 1025u);
}

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoLoadAddImm_ExtractFields)
{
    MockXnQuerySuccess();

    Variable src{&context};
    Variable srcOffset{&context};
    Variable dst{&context};
    auto rep = std::make_shared<CcuRepLoadAddImm>(&insGen, src, 2, srcOffset, 0x55, dst);

    std::vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenErrorInfoLoadAddImm(MakeBaseInfo(), rep, errorInfo);

    ASSERT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::LOAD_ADD_IMM);
    EXPECT_EQ(errorInfo[0].msg.loadAddImm.srcId, src.Id());
    EXPECT_EQ(errorInfo[0].msg.loadAddImm.srcOffsetId, srcOffset.Id());
    EXPECT_EQ(errorInfo[0].msg.loadAddImm.dstId, dst.Id());
    EXPECT_EQ(errorInfo[0].msg.loadAddImm.immAddValue, 0x55);
    EXPECT_EQ(errorInfo[0].msg.loadAddImm.srcValue, 0u);
    EXPECT_EQ(errorInfo[0].msg.loadAddImm.srcOffsetValue, 0u);
}

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoStoreAddImm_ExtractFields)
{
    MockXnQuerySuccess();

    Variable dst{&context};
    Variable dstOffset{&context};
    Variable src{&context};
    auto rep = std::make_shared<CcuRepStoreAddImm>(&insGen, dst, 3, dstOffset, 0x66, src);

    std::vector<CcuErrorInfo> errorInfo;
    CcuTaskException::GenErrorInfoStoreAddImm(MakeBaseInfo(), rep, errorInfo);

    ASSERT_EQ(errorInfo.size(), 1u);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::STORE_ADD_IMM);
    EXPECT_EQ(errorInfo[0].msg.storeAddImm.dstId, dst.Id());
    EXPECT_EQ(errorInfo[0].msg.storeAddImm.dstOffsetId, dstOffset.Id());
    EXPECT_EQ(errorInfo[0].msg.storeAddImm.srcId, src.Id());
    EXPECT_EQ(errorInfo[0].msg.storeAddImm.immAddValue, 0x66);
    EXPECT_EQ(errorInfo[0].msg.storeAddImm.dstValue, 0u);
    EXPECT_EQ(errorInfo[0].msg.storeAddImm.dstOffsetValue, 0u);
}

/* ---------------- GenErrorInfoByRepType 分发: 6 个新类型 ---------------- */

TEST_F(CcuTaskExceptionHalfRttTest, GenErrorInfoByRepType_DispatchAllHalfRttTypes)
{
    MockXnQuerySuccess();

    Variable channel{&context};
    Address addr;
    Variable token;
    RemoteAddr varAddr{addr, token};
    Variable targetValue{&context};
    Variable src{&context};
    Variable dst{&context};
    Variable srcOffset{&context};
    CompletedEvent sem{&context};

    std::vector<std::shared_ptr<CcuRepBase>> repList = {
        std::make_shared<CcuRepWriteVarAtomic>(&insGen, channel, varAddr, targetValue, sem, 0xF),
        std::make_shared<CcuRepLoadAddImm>(&insGen, src, 2, srcOffset, 0x55, dst),
    };

    for (auto& repBase : repList) {
        std::vector<CcuErrorInfo> errorInfo;
        CcuTaskException::GenErrorInfoByRepType(MakeBaseInfo(), repBase, errorInfo);
        ASSERT_EQ(errorInfo.size(), 1u);
        // 分发后 errorType 与 repType 对应(不走 DEFAULT)
        EXPECT_NE(errorInfo[0].type, CcuErrorType::DEFAULT);
        EXPECT_EQ(errorInfo[0].instrId, repBase->StartInstrId());
    }
}
