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

#include "ccu_microcode_common_v1.h"
#include "ccu_resource_manager.h"
#include "ccu_resource_v1.h"

class CcuResourceV1Test : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// Test: Constructor with valid parameters
TEST_F(CcuResourceV1Test, ConstructorValidParams) { EXPECT_NO_THROW(CcuResourceV1 resource(0, 4)); }

// Test: Constructor with different rank IDs
TEST_F(CcuResourceV1Test, ConstructorDifferentRankIds)
{
    for (int rankId = 0; rankId < 8; rankId++) {
        EXPECT_NO_THROW(CcuResourceV1 resource(rankId, 8));
    }
}

// Test: Constructor with different rank sizes
TEST_F(CcuResourceV1Test, ConstructorDifferentRankSizes)
{
    uint32_t rankSizes[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (auto rankSize : rankSizes) {
        EXPECT_NO_THROW(CcuResourceV1 resource(0, rankSize));
    }
}

// Test: Reset method
TEST_F(CcuResourceV1Test, ResetMethod)
{
    CcuResourceV1 resource(0, 4);
    EXPECT_NO_THROW(resource.Reset());
}

// Test: Default destructor
TEST_F(CcuResourceV1Test, DefaultDestructor)
{
    CcuResourceV1* resource = new CcuResourceV1(0, 4);
    EXPECT_NO_THROW(delete resource);
}

// Test: Resource size constants
TEST_F(CcuResourceV1Test, ResourceSizeConstants)
{
    EXPECT_GT(SimCcuV1::CCU_RESOURCE_XN_MAX, 0);
    EXPECT_GT(SimCcuV1::CCU_RESOURCE_GSA_MAX, 0);
    EXPECT_GT(SimCcuV1::CCU_RESOURCE_CKE_MAX, 0);
    EXPECT_GT(SimCcuV1::CCU_RESOURCE_MS_NUM, 0);
    EXPECT_GT(SimCcuV1::CCU_RESOURCE_MS_SIZE, 0);
}

// Test: MS size calculation
TEST_F(CcuResourceV1Test, MsSizeCalculation)
{
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_MS_SIZE, SimCcuV1::CCU_RESOURCE_MS_NUM * HcclSim::BYTE_NUM_4K);
}

// Test: Channel number constant
TEST_F(CcuResourceV1Test, ChannelNumConstant) { EXPECT_EQ(SimCcuV1::MAX_CCU_CHANNEL_NUM, 128); }

// Test: Instruction type constants
TEST_F(CcuResourceV1Test, InstructionTypeConstants)
{
    EXPECT_EQ(SimCcuV1::LOAD_TYPE, 0x0);
    EXPECT_EQ(SimCcuV1::CTRL_TYPE, 0x1);
    EXPECT_EQ(SimCcuV1::TRANS_TYPE, 0x2);
    EXPECT_EQ(SimCcuV1::REDUCE_TYPE, 0x3);
}

// Test: Load instruction code constants
TEST_F(CcuResourceV1Test, LoadInstrCodeConstants)
{
    EXPECT_EQ(SimCcuV1::LOADSQEARGSTOGSA_CODE, 0x0);
    EXPECT_EQ(SimCcuV1::LOADSQEARGSTOXN_CODE, 0x1);
    EXPECT_EQ(SimCcuV1::LOADIMDTOGSA_CODE, 0x2);
    EXPECT_EQ(SimCcuV1::LOADIMDTOXN_CODE, 0x3);
    EXPECT_EQ(SimCcuV1::LOADGSAXN_CODE, 0x4);
    EXPECT_EQ(SimCcuV1::LOADGSAGSA_CODE, 0x5);
    EXPECT_EQ(SimCcuV1::LOADXX_CODE, 0x6);
}

// Test: Control instruction code constants
TEST_F(CcuResourceV1Test, CtrlInstrCodeConstants)
{
    EXPECT_EQ(SimCcuV1::LOOP_CODE, 0x0);
    EXPECT_EQ(SimCcuV1::LOOPGROUP_CODE, 0x1);
    EXPECT_EQ(SimCcuV1::SETCKE_CODE, 0x2);
    EXPECT_EQ(SimCcuV1::CLEARCKE_CODE, 0x4);
    EXPECT_EQ(SimCcuV1::JMP_CODE, 0x5);
}

// Test: Transfer instruction code constants
TEST_F(CcuResourceV1Test, TransInstrCodeConstants)
{
    EXPECT_EQ(SimCcuV1::TRANSLOCMEMTOLOCMS_CODE, 0x0);
    EXPECT_EQ(SimCcuV1::TRANSRMTMEMTOLOCMS_CODE, 0x1);
    EXPECT_EQ(SimCcuV1::TRANSLOCMSTOLOCMEM_CODE, 0x2);
    EXPECT_EQ(SimCcuV1::TRANSLOCMSTORMTMEM_CODE, 0x3);
    EXPECT_EQ(SimCcuV1::TRANSRMTMSTOLOCMEM_CODE, 0x4);
    EXPECT_EQ(SimCcuV1::TRANSLOCMSTOLOCMS_CODE, 0x5);
    EXPECT_EQ(SimCcuV1::TRANSRMTMSTOLOCMS_CODE, 0x6);
    EXPECT_EQ(SimCcuV1::TRANSLOCMSTORMTMS_CODE, 0x7);
    EXPECT_EQ(SimCcuV1::TRANSRMTMEMTOLOCMEM_CODE, 0x8);
    EXPECT_EQ(SimCcuV1::TRANSLOCMEMTORMTMEM_CODE, 0x9);
    EXPECT_EQ(SimCcuV1::TRANSLOCMEMTOLOCMEM_CODE, 0xa);
    EXPECT_EQ(SimCcuV1::SYNCCKE_CODE, 0xb);
    EXPECT_EQ(SimCcuV1::SYNCGSA_CODE, 0xc);
    EXPECT_EQ(SimCcuV1::SYNCXN_CODE, 0xd);
}

// Test: Reduce instruction code constants
TEST_F(CcuResourceV1Test, ReduceInstrCodeConstants)
{
    EXPECT_EQ(SimCcuV1::ADD_CODE, 0x0);
    EXPECT_EQ(SimCcuV1::MAX_CODE, 0x1);
    EXPECT_EQ(SimCcuV1::MIN_CODE, 0x2);
}

// Test: Invalid instr start index
TEST_F(CcuResourceV1Test, InvalidInstrStartIndex) { EXPECT_EQ(SimCcuV1::INVALID_INSTR_START_INDEX, 0xFFFF); }

// Test: Loop engine resource
TEST_F(CcuResourceV1Test, LoopEngineResource)
{
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_LOOP_ENGINE_NUM, 192);
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_LOOP_ENGINE_MAX, 200);
}

// Test: Memory slice 4K constant
TEST_F(CcuResourceV1Test, MemSlice4KConstant) { EXPECT_EQ(SimCcuV1::CCU_RESOURCE_MEM_SLICE_4K, 4096); }

// Test: XN resource limits
TEST_F(CcuResourceV1Test, XnResourceLimits)
{
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_XN_NUM, 3024);
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_XN_MAX, 3072);
}

// Test: GSA resource limits
TEST_F(CcuResourceV1Test, GsaResourceLimits)
{
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_GSA_NUM, 3024);
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_GSA_MAX, 3072);
}

// Test: CKE resource limits
TEST_F(CcuResourceV1Test, CkeResourceLimits)
{
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_CKE_NUM, 1008);
    EXPECT_EQ(SimCcuV1::CCU_RESOURCE_CKE_MAX, 1024);
}

// Test: Multiple instances
TEST_F(CcuResourceV1Test, MultipleInstances)
{
    CcuResourceV1 resource1(0, 4);
    CcuResourceV1 resource2(1, 4);
    CcuResourceV1 resource3(2, 4);

    EXPECT_NO_THROW(resource1.Reset());
    EXPECT_NO_THROW(resource2.Reset());
    EXPECT_NO_THROW(resource3.Reset());
}

// Test: Boundary rank ID
TEST_F(CcuResourceV1Test, BoundaryRankId)
{
    EXPECT_NO_THROW(CcuResourceV1 resource(0, 128));
    EXPECT_NO_THROW(CcuResourceV1 resource(127, 128));
}

// Test: Boundary rank size
TEST_F(CcuResourceV1Test, BoundaryRankSize)
{
    EXPECT_NO_THROW(CcuResourceV1 resource(0, 1));
    EXPECT_NO_THROW(CcuResourceV1 resource(0, 256));
}

// Test: Resource creation through manager
TEST_F(CcuResourceV1Test, ResourceThroughManager)
{
    CcuResourceManager& mgr = CcuResourceManager::GetInstance();
    EXPECT_NO_THROW(mgr.Init(0, 4, RunnerCcuVersion::CCU_V1, {}));
}

// Test: DIE_NUM constant
TEST_F(CcuResourceV1Test, DieNumConstant) { EXPECT_EQ(HcclSim::DIE_NUM, 2); }

// Test: BYTE_NUM_4K constant
TEST_F(CcuResourceV1Test, ByteNum4KConstant) { EXPECT_EQ(HcclSim::BYTE_NUM_4K, 4096); }
