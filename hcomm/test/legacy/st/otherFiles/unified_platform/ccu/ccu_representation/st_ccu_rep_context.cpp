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

#include <vector>
#include <climits>

#include "ccu_datatype.h"
#include "ccu_rep_context.h"
#include "ccu_context_resource.h"
#include "ccu_rep.h"
#include "log.h"
using namespace Hccl;
using namespace CcuRep;

class CcuRepCtxTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuRepCtxTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuRepCtxTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuRepCtxTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuRepCtxTest TearDown" << std::endl;
    }
};

TEST_F(CcuRepCtxTest, RepNopTest)
{
    CcuRepNop nopRep;
    EXPECT_EQ(nopRep.Type(), CcuRepType::NOP);
    EXPECT_EQ(nopRep.Translated(), false);
    EXPECT_EQ(nopRep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    nopRep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(nopRep.Translated(), true);
    EXPECT_EQ(nopRep.StartInstrId(), 3);
    HCCL_INFO("%s", nopRep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBlockTest)
{
    CcuRepBlock blockRep("block");
    blockRep.Append(std::make_shared<CcuRepNop>());
    EXPECT_EQ(blockRep.Type(), CcuRepType::BLOCK);
    EXPECT_EQ(blockRep.GetLabel(), "block");
    CcuInstr instr[10];
    CcuInstr* instrPtr = instr;
    uint16_t instrId = 10;
    TransDep dep;
    blockRep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(blockRep.Translated(), true);
    EXPECT_EQ(blockRep.StartInstrId(), 10);
    HCCL_INFO("%s", blockRep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepLoadArgTest)
{
    Variable var;
    CcuRepLoadArg rep(var, 0);
    EXPECT_EQ(rep.Type(), CcuRepType::LOAD_ARG);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBufLocReadTest)
{
    Memory src;
    CcuRep::CcuBuffer dst;
    Variable len;
    MaskSignal sig;

    CcuRepBufLocRead rep(src, dst, len, sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::BUF_LOC_READ);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBufLocWriteTest)
{
    CcuRep::CcuBuffer src;
    Memory dst;
    Variable len;
    MaskSignal sig;

    CcuRepBufLocWrite rep(src, dst, len, sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::BUF_LOC_WRITE);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);
    rep.Describe();

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBufLocReduceTest)
{
    std::vector<CcuRep::CcuBuffer> bufs(4);
    MaskSignal sig;
    Variable len;

    CcuRepBufReduce rep(bufs, 4, 0, 0, 0, sig, len, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::BUF_REDUCE);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBufLocReduceTest_fp16)
{
    std::vector<CcuRep::CcuBuffer> bufs(4);
    MaskSignal sig;
    Variable len;

    CcuRepBufReduce rep(bufs, 4, 0, 1, 0, sig, len, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::BUF_REDUCE);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBufLocReduceTest_bf16)
{
    std::vector<CcuRep::CcuBuffer> bufs(4);
    MaskSignal sig;
    Variable len;

    CcuRepBufReduce rep(bufs, 4, 0, 2, 0, sig, len, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::BUF_REDUCE);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBufLocReduceTest_max)
{
    std::vector<CcuRep::CcuBuffer> bufs(4);
    MaskSignal sig;
    Variable len;

    CcuRepBufReduce rep(bufs, 4, 0, 0, 1, sig, len, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::BUF_REDUCE);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepBufLocReduceTest_min)
{
    std::vector<CcuRep::CcuBuffer> bufs(4);
    MaskSignal sig;
    Variable len;

    CcuRepBufReduce rep(bufs, 4, 0, 0, 2, sig, len, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::BUF_REDUCE);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepLocCpyTest)
{
    Memory src;
    Memory dst;
    Variable len;
    MaskSignal sig;

    {
        CcuRepLocCpy rep(dst, src, len, sig, 1);
        EXPECT_EQ(rep.Type(), CcuRepType::LOCAL_CPY);
        EXPECT_EQ(rep.Translated(), false);
        EXPECT_EQ(rep.StartInstrId(), 0);

        CcuInstr instr;
        CcuInstr* instrPtr = &instr;
        uint16_t instrId = 3;
        TransDep dep;
        rep.Translate(instrPtr, instrId, dep);
        EXPECT_EQ(rep.Translated(), true);
        EXPECT_EQ(rep.StartInstrId(), 3);
        HCCL_INFO("%s", rep.Describe().c_str());
    }
    {
        CcuRepLocCpy rep(dst, src, len, 0, 0, sig, 1);
        EXPECT_EQ(rep.Type(), CcuRepType::LOCAL_REDUCE);
        EXPECT_EQ(rep.Translated(), false);
        EXPECT_EQ(rep.StartInstrId(), 0);

        CcuInstr instr;
        CcuInstr* instrPtr = &instr;
        uint16_t instrId = 3;
        TransDep dep;
        rep.Translate(instrPtr, instrId, dep);
        EXPECT_EQ(rep.Translated(), true);
        EXPECT_EQ(rep.StartInstrId(), 3);
        HCCL_INFO("%s", rep.Describe().c_str());
    }
}

TEST_F(CcuRepCtxTest, RepLocPostSemTest)
{
    MaskSignal sig;

    CcuRepLocPostSem rep(sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::LOC_POST_SEM);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepLocPostSem_Translate_fail_test)
{
    MaskSignal sig;

    CcuRepLocPostSem rep(sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::LOC_POST_SEM);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = USHRT_MAX;
    TransDep dep;
    EXPECT_THROW(rep.Translate(instrPtr, instrId, dep), InternalException);
}

TEST_F(CcuRepCtxTest, RepLocWaitSemTest)
{
    MaskSignal sig;

    CcuRepLocWaitSem rep(sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::LOC_WAIT_SEM);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepLocWaitSem_Translate_fail_test)
{
    MaskSignal sig;

    CcuRepLocWaitSem rep(sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::LOC_WAIT_SEM);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = USHRT_MAX;
    TransDep dep;
    EXPECT_THROW(rep.Translate(instrPtr, instrId, dep), InternalException);
}

TEST_F(CcuRepCtxTest, RepPostSharedSemTest)
{
    MaskSignal sig;

    CcuRepPostSharedSem rep(sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::POST_SHARED_SEM);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepPostSharedVarTest)
{
    Variable srcVar;
    Variable dstVar;
    MaskSignal sig;

    CcuRepPostSharedVar rep(srcVar, dstVar, sig, 1);
    EXPECT_EQ(rep.Type(), CcuRepType::POST_SHARED_VAR);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr;
    CcuInstr* instrPtr = &instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepSetLoopTest)
{
    Variable loopParam;
    Executor exec;
    Variable var;

    CcuRepSetLoop rep(loopParam, exec, var);
    EXPECT_EQ(rep.Type(), CcuRepType::SET_LOOP);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr[10];
    CcuInstr* instrPtr = instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepSetLoop_Translate_fail_test)
{
    Variable loopParam;
    Executor exec;
    Variable var;

    CcuRepSetLoop rep(loopParam, exec, var);
    EXPECT_EQ(rep.Type(), CcuRepType::SET_LOOP);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr[10];
    CcuInstr* instrPtr = instr;
    uint16_t instrId = USHRT_MAX;
    TransDep dep;
    EXPECT_THROW(rep.Translate(instrPtr, instrId, dep), InternalException);
}

TEST_F(CcuRepCtxTest, RepJumpLabelTest)
{
    CcuRepJumpLabel rep("test");
    EXPECT_EQ(rep.Type(), CcuRepType::JUMP_LABEL);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr[10];
    CcuInstr* instrPtr = instr;
    uint16_t instrId = 3;
    TransDep dep;
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 3);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepJumpTest)
{
    CcuRepJump rep("test", Variable());
    auto refRep = std::make_shared<CcuRepJumpLabel>("test");
    rep.Reference(refRep);
    EXPECT_EQ(rep.Type(), CcuRepType::JUMP);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr[10];
    CcuInstr* instrPtr = instr;
    uint16_t instrId = 3;
    TransDep dep;
    refRep->Translate(instrPtr, instrId, dep);
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 4);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepJumpEQTest)
{
    Variable cond;
    CcuRepJumpEQ rep("test", Variable(), cond, 1);
    auto refRep = std::make_shared<CcuRepJumpLabel>("test");
    rep.Reference(refRep);
    EXPECT_EQ(rep.Type(), CcuRepType::JUMP_EQ);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr[10];
    CcuInstr* instrPtr = instr;
    uint16_t instrId = 3;
    TransDep dep;
    refRep->Translate(instrPtr, instrId, dep);
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 4);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepJumpNETest)
{
    Variable cond;
    CcuRepJumpNE rep("test", Variable(), cond, 1);
    auto refRep = std::make_shared<CcuRepJumpLabel>("test");
    rep.Reference(refRep);
    EXPECT_EQ(rep.Type(), CcuRepType::JUMP_NE);
    EXPECT_EQ(rep.Translated(), false);
    EXPECT_EQ(rep.StartInstrId(), 0);

    CcuInstr instr[10];
    CcuInstr* instrPtr = instr;
    uint16_t instrId = 3;
    TransDep dep;
    refRep->Translate(instrPtr, instrId, dep);
    rep.Translate(instrPtr, instrId, dep);
    EXPECT_EQ(rep.Translated(), true);
    EXPECT_EQ(rep.StartInstrId(), 4);
    HCCL_INFO("%s", rep.Describe().c_str());
}

TEST_F(CcuRepCtxTest, RepCtxTest)
{
    CcuRepContext ctx;
    auto nopRep = std::make_shared<CcuRepNop>();
    auto blockRep = std::make_shared<CcuRepBlock>("block");
    EXPECT_EQ(blockRep->GetLabel(), "block");
    auto mainBlock = ctx.CurrentBlock();
    ctx.SetCurrentBlock(blockRep);
    ctx.Append(nopRep);

    EXPECT_EQ(blockRep->InstrCount(), 1);
    auto repInBlock = blockRep->GetReps();
    EXPECT_EQ(repInBlock.size(), 1);

    ctx.SetCurrentBlock(mainBlock);
    ctx.Append(blockRep);
    auto reps = ctx.GetRepSequence();
    ctx.DumpReprestation();

    ctx.SetDieId(1);
    EXPECT_EQ(ctx.GetDieId(), 1);
    ctx.SetMissionId(2);
    EXPECT_EQ(ctx.GetMissionId(), 2);
    ctx.SetMissionKey(4);
    EXPECT_EQ(ctx.GetMissionKey(), 4);
}

TEST_F(CcuRepCtxTest, DataTypeTest)
{
    Variable varA;

    varA.Reset(200, 3);
    EXPECT_EQ(varA.Id(), 200);
    EXPECT_EQ(varA.DieId(), 3);

    varA.Reset(100);
    varA.SetDieId(2);
    EXPECT_EQ(varA.Id(), 100);
    EXPECT_EQ(varA.DieId(), 2);
}
