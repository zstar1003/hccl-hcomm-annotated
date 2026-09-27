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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "prim_translator.h"
#include "communicator_impl.h"
#include "hccp_tlv_hdc_manager.h"
using namespace Hccl;
using namespace std;
class PrimTranslatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PrimTranslatorTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PrimTranslatorTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        communicator = new CommunicatorImpl();
        masterPrimQue = make_shared<PrimQueue>();
        DataSlice srcSlice(BufferType::INPUT, 0, 100);
        DataSlice dstSlice(BufferType::SCRATCH, 0, 100);
        unique_ptr<PrimLocalCopy> localCopy = make_unique<PrimLocalCopy>(srcSlice, dstSlice);

        masterPrimQue->Append(std::move(localCopy));
        std::cout << "A Test case in PrimTranslatorTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete communicator;
        std::cout << "A Test case in PrimTranslatorTest TearDown" << std::endl;
    }

    shared_ptr<PrimQueue> masterPrimQue;
    CommunicatorImpl* communicator;
};

TEST_F(PrimTranslatorTest, test_Translate_one_queue)
{
    HccpTlvHdcManager::GetInstance();
    PrimTranslator primTranslator;

    shared_ptr<InsQueue> masterInsQue;
    EXPECT_EQ(nullptr, masterInsQue.get());

    masterInsQue = primTranslator.Translate(*masterPrimQue.get());

    EXPECT_NE(nullptr, masterInsQue.get());
    EXPECT_EQ(0, masterInsQue->SizeOfSlaves());
    EXPECT_EQ(masterPrimQue->GetId(), masterInsQue->GetId());
}

TEST_F(PrimTranslatorTest, test_Translate_many_queue)
{
    PrimTranslator primTranslator;

    shared_ptr<InsQueue> masterInsQue;
    EXPECT_EQ(nullptr, masterInsQue.get());

    shared_ptr<PrimQueue> slavePrimQue;
    slavePrimQue = masterPrimQue->Fork();

    masterInsQue = primTranslator.Translate(*masterPrimQue.get());

    EXPECT_NE(nullptr, masterInsQue.get());
    EXPECT_EQ(1, masterInsQue->SizeOfSlaves());
}

TEST_F(PrimTranslatorTest, test_translate_multiple_prim_queue)
{
    // Given master, slave
    PrimTranslator primTranslator;
    u32 notifyIdx = 0;
    DataSlice slaveSrcSlice(BufferType::OUTPUT, 0, 100);
    DataSlice slaveDstSlice(BufferType::OUTPUT, 100, 100);
    unique_ptr<PrimLocalCopy> localCopySlave = make_unique<PrimLocalCopy>(slaveSrcSlice, slaveDstSlice);
    shared_ptr<PrimQueue> slavePrimQue = masterPrimQue->Fork();
    slavePrimQue->Append(std::move(localCopySlave));

    // When
    shared_ptr<InsQueue> masterInsQue = primTranslator.Translate(*masterPrimQue.get());
    string slaveRes;
    for (auto iter = masterInsQue->IterSlaves(); iter.HasNext(); ++iter) {
        slaveRes = iter->First()->Describe();
    }
    string masterRes = masterInsQue->First()->Describe();

    // Then
    vector<string> insStringVec(2);
    DataSlice srcSlice(BufferType::INPUT, 0, 100);
    DataSlice dstSlice(BufferType::SCRATCH, 0, 100);
    InsLocalCopy insLocalCopy(srcSlice, dstSlice);
    insStringVec[0] = insLocalCopy.Describe();
    EXPECT_EQ(insStringVec[0], masterRes);

    InsLocalCopy insLocalCopySlave(slaveSrcSlice, slaveDstSlice);
    insStringVec[1] = insLocalCopySlave.Describe();
    EXPECT_EQ(insStringVec[1], slaveRes);
}
