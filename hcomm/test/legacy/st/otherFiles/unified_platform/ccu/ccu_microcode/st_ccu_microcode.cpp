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
using namespace Hccl;
using namespace CcuRep;

class CcuMicroCodeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuMicroCodeTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuMicroCodeTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuMicroCodeTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuMicroCodeTest TearDown" << std::endl;
    }
};

TEST_F(CcuMicroCodeTest, Test)
{
    CcuInstr ccuInstr[100];
    uint32_t index = 0;

    LoadSqeArgsToGSAInstr(ccuInstr + index++, 0, 0);
    LoadSqeArgsToXnInstr(ccuInstr + index++, 0, 0);
    LoadImdToGSAInstr(ccuInstr + index++, 0, 10llu);
    LoadImdToXnInstr(ccuInstr + index++, 0, 20llu);
    LoadGSAXnInstr(ccuInstr + index++, 0, 1, 2);
    LoadGSAGSAInstr(ccuInstr + index++, 0, 1, 2);
    LoadXXInstr(ccuInstr + index++, 0, 1, 2);

    LoopInstr(ccuInstr + index++, 0, 0, 1);
    LoopGroupInstr(ccuInstr + index++, 0, 1, 2, 0);
    SetCKEInstr(ccuInstr + index++, 0, 0x1, 0, 0xff, 1);
    ClearCKEInstr(ccuInstr + index++, 0, 0x1, 0, 0xff, 1);
    JumpInstr(ccuInstr + index++, 0, 10, 100llu);

    TransLocMemToLocMSInstr(ccuInstr + index++, 0, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);
    TransRmtMemToLocMSInstr(ccuInstr + index++, 0, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);
    TransLocMSToLocMemInstr(ccuInstr + index++, 0, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);
    TransLocMSToRmtMemInstr(ccuInstr + index++, 0, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);

    TransRmtMSToLocMemInstr(ccuInstr + index++, 0, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);

    TransLocMSToLocMSInstr(ccuInstr + index++, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);
    TransRmtMSToLocMSInstr(ccuInstr + index++, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);
    TransLocMSToRmtMSInstr(ccuInstr + index++, 0, 0, 1, 0, 0, 0x2, 0, 0x1, 0, 0xff, 1, 0);

    TransRmtMemToLocMemInstr(ccuInstr + index++, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0x1, 0, 0xff, 1, 0, 0);
    TransLocMemToRmtMemInstr(ccuInstr + index++, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0x1, 0, 0xff, 1, 0, 0);
    TransLocMemToLocMemInstr(ccuInstr + index++, 0, 0, 0, 0, 1, 0, 0, 0x1, 0, 0xff, 1, 0);

    SyncCKEInstr(ccuInstr + index++, 0, 0, 0x1, 0, 0, 0x1, 0, 0xff, 1);
    SyncGSAInstr(ccuInstr + index++, 0, 0, 0, 0, 0x2, 0, 0x1, 0, 0xff, 1);
    SyncXnInstr(ccuInstr + index++, 0, 0, 0, 0, 0x2, 0, 0x1, 0, 0xff, 1);

    uint16_t MS[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    AddInstr(ccuInstr + index++, MS, 6, 0, 0, 0, 0x1, 0, 0xff, 1, 0);
    MaxInstr(ccuInstr + index++, MS, 6, 0, 0, 0x1, 0, 0xff, 1, 0);
    MinInstr(ccuInstr + index++, MS, 6, 0, 0, 0x1, 0, 0xff, 1, 0);

    std::vector<std::string> instrStr;
    for (int i = 0; i < index; i++) {
        instrStr.emplace_back(ParseInstr(ccuInstr + i));
    }

    for (int i = 0; i < instrStr.size(); i++) {
        HCCL_INFO("index[%d]: %s", i, instrStr[i].c_str());
    }
}
