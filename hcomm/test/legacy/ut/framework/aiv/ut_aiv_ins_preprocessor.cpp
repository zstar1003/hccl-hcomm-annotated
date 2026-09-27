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
#define private public
#define protected public

#include "ccu_instruction_all_gather_mesh1d.h"
#include "aiv_ins_preprocessor.h"
#include "coll_operator.h"
#include "local_rma_buf_manager.h"
#include "rma_conn_manager.h"
#include "virtual_topo.h"

#undef private
#undef protected

using namespace Hccl;

using namespace std;

class AivInsPreprocessorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AivInsPreprocessorTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AivInsPreprocessorTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AivInsPreprocessorTest SetUP" << std::endl; }

    virtual void TearDown() { std::cout << "A Test case in AivInsPreprocessorTest TearDown" << std::endl; }
};

TEST_F(AivInsPreprocessorTest, should_continue_when_calling_preprocess_not_aivIns)
{
    CommunicatorImpl* comm;
    AivInsPreprocessor preprocessor(comm);
    auto insQueue = make_shared<InsQueue>();

    std::unique_ptr<CcuInstruction> ccuIns = std::make_unique<CcuInstructionAllGatherMesh1D>();
    insQueue->Append(std::move(ccuIns));
    // check
    EXPECT_NO_THROW(preprocessor.Preprocess(insQueue));
}

// 覆盖 BatchBuildUrmaTransports 入口的 fake Reg 改动：opTag 取值、aivUrmaBufferTag 隔离标签、空 links 跳过循环
TEST_F(AivInsPreprocessorTest, batch_build_urma_transports_empty_links_covers_fake_reg_opTag)
{
    CommunicatorImpl comm;
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "test_op";
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    // memTransportManager 保持 nullptr，触发 CHECK_NULLPTR 抛异常，提前结束以避免 while 等待

    AivInsPreprocessor preprocessor(&comm);
    std::vector<LinkData> emptyLinks;
    EXPECT_ANY_THROW(preprocessor.BatchBuildUrmaTransports(emptyLinks));
}
