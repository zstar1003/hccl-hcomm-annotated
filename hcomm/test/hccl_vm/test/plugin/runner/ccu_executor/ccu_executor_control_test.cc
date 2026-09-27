/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

#include "ccu_executor_base.h"
#include "ccu_microcode_common_v1.h"
#include "clear_cke_executor.h"
#include "jump_executor.h"
#include "loop_executor.h"
#include "loop_group_executor.h"
#include "set_cke_executor.h"

using namespace hcomm::CcuRep;

// Base fixture with common helpers
class ControlExecutorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// ClearCkeExecutor Tests
// ============================================================================
class ClearCkeExecutorTest : public ControlExecutorTest {};

// Test: ClearCkeExecutor struct size check
TEST_F(ClearCkeExecutorTest, StructSize) { EXPECT_GT(sizeof(ClearCkeExecutor), 0); }

// Test: ClearCkeExecutor default constructor
TEST_F(ClearCkeExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    ClearCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor parameterized constructor with valid parameters
TEST_F(ClearCkeExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor Parser with zero values
TEST_F(ClearCkeExecutorTest, ParserZeroValues)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor Parser with max values
TEST_F(ClearCkeExecutorTest, ParserMaxValues)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: ClearCkeExecutor Parser with boundary values
TEST_F(ClearCkeExecutorTest, ParserBoundaryValues)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    // Set specific boundary values
    instr.v1.clearCKE.clearType = 1;
    instr.v1.clearCKE.clearCKEId = 0xFFFF;
    instr.v1.clearCKE.clearMask = 0xFFFF;
    instr.v1.clearCKE.waitCKEId = 0xFFFF;
    instr.v1.clearCKE.waitCKEMask = 0xFFFF;

    ClearCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: ClearCkeExecutor with different rank and die IDs
TEST_F(ClearCkeExecutorTest, DifferentRankDieIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (int rankId = 0; rankId < 8; rankId++) {
        for (int dieId = 0; dieId < 4; dieId++) {
            ClearCkeExecutor executor(0, rankId, dieId, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// Test: ClearCkeExecutor Describe returns non-empty string
TEST_F(ClearCkeExecutorTest, DescribeNonEmpty)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.clearCKE.clearType = 1;
    instr.v1.clearCKE.clearCKEId = 100;
    instr.v1.clearCKE.clearMask = 0x00FF;
    instr.v1.clearCKE.waitCKEId = 50;
    instr.v1.clearCKE.waitCKEMask = 0xFF00;

    ClearCkeExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Clear CKE"), std::string::npos);
}

// Test: ClearCkeExecutor inheritance from CcuExecutorBase
TEST_F(ClearCkeExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    ClearCkeExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// ============================================================================
// JumpExecutor Tests
// ============================================================================
class JumpExecutorTest : public ControlExecutorTest {};

// Test: JumpExecutor struct size check
TEST_F(JumpExecutorTest, StructSize) { EXPECT_GT(sizeof(JumpExecutor), 0); }

// Test: JumpExecutor default constructor
TEST_F(JumpExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor parameterized constructor
TEST_F(JumpExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    JumpExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor Parser with zero values
TEST_F(JumpExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor Parser with max values
TEST_F(JumpExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: JumpExecutor Parser with boundary values
TEST_F(JumpExecutorTest, ParserBoundaryValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.jmp.dstInstrXnId = 0xFFFF;
    instr.v1.jmp.conditionXnId = 0xFFFF;
    instr.v1.jmp.expectData = 0xFFFFFFFFFFFFFFFFULL;

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: JumpExecutor with different expectData values
TEST_F(JumpExecutorTest, DifferentExpectDataValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint64_t testValues[] = {0, 1, 0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFULL, 1000, 0x12345678ABCDEF00ULL};

    for (auto val : testValues) {
        instr.v1.jmp.expectData = val;
        JumpExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: JumpExecutor Describe contains expected keywords
TEST_F(JumpExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.jmp.dstInstrXnId = 100;
    instr.v1.jmp.conditionXnId = 50;
    instr.v1.jmp.expectData = 12345;

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Jump"), std::string::npos);
}

// Test: JumpExecutor with different stream IDs
TEST_F(JumpExecutorTest, DifferentStreamIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (int streamId = 0; streamId < 16; streamId++) {
        JumpExecutor executor(streamId, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: JumpExecutor inheritance check
TEST_F(JumpExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: JumpExecutor multiple parse calls
TEST_F(JumpExecutorTest, MultipleParseCalls)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.jmp.dstInstrXnId = 100;

    JumpExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Parser());
}

// ============================================================================
// LoopExecutor Tests
// ============================================================================
class LoopExecutorTest : public ControlExecutorTest {};

// Test: LoopExecutor struct size check
TEST_F(LoopExecutorTest, StructSize) { EXPECT_GT(sizeof(LoopExecutor), 0); }

// Test: LoopExecutor default constructor
TEST_F(LoopExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor parameterized constructor
TEST_F(LoopExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor Parser with zero values
TEST_F(LoopExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor Parser with max values
TEST_F(LoopExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopExecutor Parser with specific loop parameters
TEST_F(LoopExecutorTest, ParserLoopParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loop.startInstrId = 10;
    instr.v1.loop.endInstrId = 100;
    instr.v1.loop.xnId = 5;

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Loop"), std::string::npos);
}

// Test: LoopExecutor with different instruction ranges
TEST_F(LoopExecutorTest, DifferentInstructionRanges)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    struct Range {
        uint16_t start;
        uint16_t end;
    };
    Range ranges[] = {{0, 10}, {10, 100}, {0, 0xFFFF}, {1000, 2000}};

    for (const auto& range : ranges) {
        instr.v1.loop.startInstrId = range.start;
        instr.v1.loop.endInstrId = range.end;

        LoopExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoopExecutor with different xn IDs
TEST_F(LoopExecutorTest, DifferentXnIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t xnIds[] = {0, 1, 100, 0x7FFF, 0xFFFF};

    for (auto xnId : xnIds) {
        instr.v1.loop.xnId = xnId;
        LoopExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoopExecutor Describe contains expected keywords
TEST_F(LoopExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loop.startInstrId = 5;
    instr.v1.loop.endInstrId = 50;
    instr.v1.loop.xnId = 10;

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("startInstrId"), std::string::npos);
    EXPECT_NE(desc.find("endInstrId"), std::string::npos);
}

// Test: LoopExecutor inheritance check
TEST_F(LoopExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: LoopExecutor with various rank and die combinations
TEST_F(LoopExecutorTest, VariousRankDieCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loop.startInstrId = 0;
    instr.v1.loop.endInstrId = 10;
    instr.v1.loop.xnId = 0;

    for (int rank = 0; rank < 4; rank++) {
        for (int die = 0; die < 2; die++) {
            LoopExecutor executor(0, rank, die, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// ============================================================================
// LoopGroupExecutor Tests
// ============================================================================
class LoopGroupExecutorTest : public ControlExecutorTest {};

// Test: LoopGroupExecutor struct size check
TEST_F(LoopGroupExecutorTest, StructSize) { EXPECT_GT(sizeof(LoopGroupExecutor), 0); }

// Test: LoopGroupExecutor default constructor
TEST_F(LoopGroupExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor parameterized constructor
TEST_F(LoopGroupExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopGroupExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor Parser with zero values
TEST_F(LoopGroupExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor Parser with max values
TEST_F(LoopGroupExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: LoopGroupExecutor Parser with specific parameters
TEST_F(LoopGroupExecutorTest, ParserSpecificParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.loopGroup.startLoopInstrId = 10;
    instr.v1.loopGroup.xnId = 5;
    instr.v1.loopGroup.xmId = 6;
    instr.v1.loopGroup.highPerfModeEn = 1;

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: LoopGroupExecutor with different highPerfModeEn values
TEST_F(LoopGroupExecutorTest, DifferentHighPerfModeValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t mode = 0; mode <= 1; mode++) {
        instr.v1.loopGroup.highPerfModeEn = mode;
        LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: LoopGroupExecutor Describe contains expected keywords
TEST_F(LoopGroupExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.loopGroup.startLoopInstrId = 5;
    instr.v1.loopGroup.xnId = 10;
    instr.v1.loopGroup.xmId = 15;
    instr.v1.loopGroup.highPerfModeEn = 1;

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("LoopGroup"), std::string::npos);
}

// Test: LoopGroupExecutor inheritance check
TEST_F(LoopGroupExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: LoopGroupExecutor with various xnId and xmId combinations
TEST_F(LoopGroupExecutorTest, VariousXnXmCombinations)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t testIds[] = {0, 1, 100, 0x7FFF, 0xFFFF};

    for (auto xnId : testIds) {
        for (auto xmId : testIds) {
            instr.v1.loopGroup.xnId = xnId;
            instr.v1.loopGroup.xmId = xmId;
            LoopGroupExecutor executor(0, 0, 0, instr, nullptr);
            EXPECT_NO_THROW(executor.Parser());
            EXPECT_NO_THROW(executor.Describe());
        }
    }
}

// ============================================================================
// SetCkeExecutor Tests
// ============================================================================
class SetCkeExecutorTest : public ControlExecutorTest {};

// Test: SetCkeExecutor struct size check
TEST_F(SetCkeExecutorTest, StructSize) { EXPECT_GT(sizeof(SetCkeExecutor), 0); }

// Test: SetCkeExecutor default constructor
TEST_F(SetCkeExecutorTest, DefaultConstructor)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor parameterized constructor
TEST_F(SetCkeExecutorTest, ParameterizedConstructor)
{
    int streamId = 0;
    int rankId = 0;
    int dieId = 0;
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SetCkeExecutor executor(streamId, rankId, dieId, instr, nullptr);
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor Parser with zero values
TEST_F(SetCkeExecutorTest, ParserZeroValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor Parser with max values
TEST_F(SetCkeExecutorTest, ParserMaxValues)
{
    CcuInstr instr;
    memset(&instr, 0xFF, sizeof(instr));

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    EXPECT_NO_THROW(executor.Describe());
}

// Test: SetCkeExecutor Parser with specific CKE parameters
TEST_F(SetCkeExecutorTest, ParserCkeParameters)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    instr.v1.setCKE.clearType = 1;
    instr.v1.setCKE.setCKEId = 100;
    instr.v1.setCKE.setCKEMask = 0xFF00;
    instr.v1.setCKE.waitCKEId = 50;
    instr.v1.setCKE.waitCKEMask = 0x00FF;

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    EXPECT_NO_THROW(executor.Parser());
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
}

// Test: SetCkeExecutor with different clearType values
TEST_F(SetCkeExecutorTest, DifferentClearTypeValues)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    for (uint16_t clearType = 0; clearType <= 2; clearType++) {
        instr.v1.setCKE.clearType = clearType;
        SetCkeExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SetCkeExecutor with different mask patterns
TEST_F(SetCkeExecutorTest, DifferentMaskPatterns)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t masks[] = {0x0000, 0x0001, 0x00FF, 0xFF00, 0xFFFF, 0x5555, 0xAAAA};

    for (auto mask : masks) {
        instr.v1.setCKE.setCKEMask = mask;
        instr.v1.setCKE.waitCKEMask = mask;
        SetCkeExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}

// Test: SetCkeExecutor Describe contains expected keywords
TEST_F(SetCkeExecutorTest, DescribeContent)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));
    instr.v1.setCKE.setCKEId = 10;
    instr.v1.setCKE.setCKEMask = 0xFF;
    instr.v1.setCKE.waitCKEId = 5;
    instr.v1.setCKE.waitCKEMask = 0xF0;

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    executor.Parser();
    std::string desc = executor.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("Set CKE"), std::string::npos);
}

// Test: SetCkeExecutor inheritance check
TEST_F(SetCkeExecutorTest, InheritanceCheck)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    SetCkeExecutor executor(0, 0, 0, instr, nullptr);
    CcuExecutorBase* base = &executor;
    EXPECT_NE(base, nullptr);
}

// Test: SetCkeExecutor with various CKE IDs
TEST_F(SetCkeExecutorTest, VariousCkeIds)
{
    CcuInstr instr;
    memset(&instr, 0, sizeof(instr));

    uint16_t ckeIds[] = {0, 1, 100, SimCcuV1::CCU_RESOURCE_CKE_NUM - 1, 0xFFFF};

    for (auto ckeId : ckeIds) {
        instr.v1.setCKE.setCKEId = ckeId;
        instr.v1.setCKE.waitCKEId = ckeId;
        SetCkeExecutor executor(0, 0, 0, instr, nullptr);
        EXPECT_NO_THROW(executor.Parser());
        EXPECT_NO_THROW(executor.Describe());
    }
}
