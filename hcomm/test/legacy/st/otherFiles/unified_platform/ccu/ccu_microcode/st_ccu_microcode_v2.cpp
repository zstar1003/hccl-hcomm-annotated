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

#include "ccu_microcode.h"
#include "log.h"
using namespace Hccl::CcuRep::CcuV2;
using namespace Hccl::CcuRep;
using namespace Hccl;
class CcuMicroCodeV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuMicroCodeV2Test tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuMicroCodeV2Test tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuMicroCodeV2Test SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuMicroCodeV2Test TearDown" << std::endl;
    }
};

TEST_F(CcuMicroCodeV2Test, Test)
{
    CcuInstr ccuInstr[100];
    uint32_t index = 0;

    LoadSqeArgsToX(ccuInstr + index++, 0, 0, 1, 0);
    LoadImdToXn(ccuInstr + index++, 0, 20llu, 1, 0);
    Nop(ccuInstr + index++);

    Assign(ccuInstr + index++, 0x1, 0x2, 1, 0);
    Add(ccuInstr + index++, 0x1, 0x2, 0x3, 1, 0);
    AddI(ccuInstr + index++, 0x1, 0x2, 0x3, 1, 0);
    Mul(ccuInstr + index++, 0x1, 0x2, 0x3, 1, 0);
    MulI(ccuInstr + index++, 0x1, 0x2, 0x3, 1, 0);
    Hccl::CcuRep::CcuV2::LoadXFromMem(ccuInstr + index++, 0x1, 0x2, 0x3, 1, CacheConfig{0, 0}, 1, 0);
    Hccl::CcuRep::CcuV2::StoreXToMem(ccuInstr + index++, 0x1, 0x2, 0x3, 1, CacheConfig{0, 0}, 1, 0);
    Hccl::CcuRep::CcuV2::Loop(ccuInstr + index++, 0, 0, 1, 1, 0);
    Hccl::CcuRep::CcuV2::LoopGroup(ccuInstr + index++, 0, 1, 2, 0);
    Hccl::CcuRep::CcuV2::SetCKE(ccuInstr + index++, 0, 0x1, 0, 0xff, 1);
    Hccl::CcuRep::CcuV2::ClearCKE(ccuInstr + index++, 0, 0x1, 0, 0xff, 1);
    Hccl::CcuRep::CcuV2::Jump(ccuInstr + index++, 0, 10, 100llu, 1);

    Hccl::CcuRep::CcuV2::TransLocMemToLocMS(ccuInstr + index++, 0, 0, 0, 1, 0, 0, 0x1, {1, 0});
    Hccl::CcuRep::CcuV2::TransLocMSToLocMem(ccuInstr + index++, 0, 0, 0, 1, 0, 0, 0xff, {1, 0});
    Hccl::CcuRep::CcuV2::TransLocMemToLocMem(ccuInstr + index++, 0, 0, 0, 0, 1, 0, 0, 0x1, {0, 0xff}, {1, 0});
    TransMem(
        ccuInstr + index++, 0, 0, 0, 0, 1, 0, Hccl::CcuRep::CcuV2::TransMemNotifyInfo{1, 1, 1},
        Hccl::CcuRep::CcuV2::TransMemReduceInfo{0, 1, 1},
        Hccl::CcuRep::CcuV2::TransMemConfig{1, 1, 1, 1, 1, 1, 1, 1, 1}, 0xff, 1);

    SyncWtX(ccuInstr + index++, 0, 0, 0x1, 0, 0, 0x1);
    SyncWtX(ccuInstr + index++, 0, 0, 0, 0, Hccl::CcuRep::CcuV2::TransMemNotifyInfo{0, 0, 0x2}, 0, 0x1);
    SyncWtX(ccuInstr + index++, Hccl::CcuRep::CcuV2::TransMemNotifyInfo{0, 0, 0x2}, 0, 0, 0);
    SyncAtX(ccuInstr + index++, 0, 0, 0, 0, 0x2, 0);

    uint16_t MS[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    ReduceAdd(ccuInstr + index++, MS, 6, 0, 0, 0, 0x1);
    ReduceMax(ccuInstr + index++, MS, 6, 0, 0, 0x1);
    ReduceMin(ccuInstr + index++, MS, 6, 0, 0, 0x1);
}
