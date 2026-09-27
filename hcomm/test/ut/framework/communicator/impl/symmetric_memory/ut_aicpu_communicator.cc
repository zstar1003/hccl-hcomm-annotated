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

#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_hccl_sqcq.h"
#include "aicpu_communicator.h"
#undef private
#undef protected
#include "llt_hccl_stub_pub.h"
#include "aicpu_hccl_process.h"
#include "symmetric_memory.h"
#include "hccl_sym_win.h"

using namespace std;
using namespace hccl;

class AicpuCommunicatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuCommunicatorTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AicpuCommunicatorTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "AicpuCommunicatorTest Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "AicpuCommunicatorTest Test TearDown" << std::endl;
    }
};

TEST_F(AicpuCommunicatorTest, Ut_PrepareSymmetricMemory_When_OpTransportResponseIsEmpty_Expect_ReturnIsHCCL_E_PARA)
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu;
    OpParam opParam;
    OpCommTransport opTransportResponse;
    HcclResult ret = hcclCommAicpu->PrepareSymmetricMemory(opParam, opTransportResponse);
    EXPECT_EQ(ret, HCCL_E_PARA);
    delete hcclCommAicpu;
}

TEST_F(AicpuCommunicatorTest, Ut_PrepareSymmetricMemory_When_LinkIsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu;
    OpParam opParam;
    OpCommTransport opTransportResponse;
    // 构造一个 level0 单元（遵循代码中使用 COMM_LEVEL0 索引）
    opTransportResponse.resize(COMM_LEVEL0 + 1);
    SingleSubCommTransport single;
    // push a nullptr link and corresponding transportRequest with isValid = true
    single.links.push_back(nullptr);
    TransportRequest req{};
    req.isValid = true;
    single.transportRequests.push_back(req);
    opTransportResponse[COMM_LEVEL0].push_back(single);

    HcclResult ret = hcclCommAicpu->PrepareSymmetricMemory(opParam, opTransportResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete hcclCommAicpu;
}

TEST_F(AicpuCommunicatorTest, Ut_ExecOp_When_SymMemEnabled_Expect_ReturnIsHCCL_SUCCESS)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    HcclComStreamInfo streamInfo;
    streamInfo.actualStreamId = 1;
    streamInfo.sqId = 1;
    streamInfo.sqDepth = 100;
    streamInfo.sqBaseAddr = &streamInfo;
    streamInfo.logicCqId = 1;
    Stream stream(streamInfo, false);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);
    hcclCommAicpu->mainStream_ = stream;
    hcclCommAicpu->retryEnable_ = true;
    hcclCommAicpu->printTaskExceptionForErr_ = true;
    hcclCommAicpu->identifier_ = "1";
    MOCKER(QuerySqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::Orchestrate).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommAicpu::PrepareSymmetricMemory).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&DeviceMem::create).stubs().with(mockcpp::any()).will(returnValue(DeviceMem()));

    std::string newTag = "tag_test_taskException";
    std::string algName = "algName_test_taskException";
    OpParam opParam;
    opParam.inputPtr = reinterpret_cast<void*>(0x1000000);
    SymmetricWindow win;
    win.stride = 1024;
    win.baseVa = reinterpret_cast<void*>(0x3000000);
    win.rankSize = 2;
    opParam.inputSymWindow = reinterpret_cast<void*>(&win);
    opParam.outputPtr = reinterpret_cast<void*>(0x2000000);
    opParam.outputSymWindow = reinterpret_cast<void*>(&win);
    HcclSymWinGetPeerPointer(opParam.inputSymWindow, opParam.inputOffset, 0, &opParam.inputPtr);
    HcclSymWinGetPeerPointer(opParam.outputSymWindow, opParam.outputOffset, 0, &opParam.outputPtr);

    hccl::AlgResourceResponse algResResponse;
    MOCKER_CPP(&HcclCommAicpu::GetAlgResponseRes).stubs().will(returnValue(HCCL_SUCCESS));
    HcclOpResParam commParam;
    commParam.localUsrRankId = 0;

    hcclCommAicpu->isSymmetricMemory_ = true;
    hcclCommAicpu->ExecOp(newTag, algName, opParam, &commParam);
}

TEST_F(AicpuCommunicatorTest, Ut_ClearOpResource_AllContainers)
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu;
    const std::string tag = "test_tag";
    const u32 rankId = 0;

    // 预置数据到 8 个容器
    hcclCommAicpu->resMap_[tag] = AlgResourceResponse();
    hcclCommAicpu->linkRes_[rankId][tag] = nullptr;
    hcclCommAicpu->linkResSio_[rankId][tag] = nullptr;
    hcclCommAicpu->linkRdmaRes_[rankId][tag].push_back(nullptr);
    hcclCommAicpu->linkRdmaResBackUp_[rankId][tag].push_back(nullptr);
    hcclCommAicpu->tagScratchMem_[tag] = nullptr;
    hcclCommAicpu->localTagResToObj_[tag].insert(0);
    hcclCommAicpu->rankTagRemoteRes_[rankId][tag] = HccltagRemoteResV3{};

    HcclResult ret = hcclCommAicpu->ClearOpResource(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证所有容器已清除
    EXPECT_EQ(hcclCommAicpu->resMap_.count(tag), 0);
    EXPECT_EQ(hcclCommAicpu->linkRes_[rankId].count(tag), 0);
    EXPECT_EQ(hcclCommAicpu->linkResSio_[rankId].count(tag), 0);
    EXPECT_EQ(hcclCommAicpu->linkRdmaRes_[rankId].count(tag), 0);
    EXPECT_EQ(hcclCommAicpu->linkRdmaResBackUp_[rankId].count(tag), 0);
    EXPECT_EQ(hcclCommAicpu->tagScratchMem_.count(tag), 0);
    EXPECT_EQ(hcclCommAicpu->localTagResToObj_.count(tag), 0);
    EXPECT_EQ(hcclCommAicpu->rankTagRemoteRes_[rankId].count(tag), 0);

    delete hcclCommAicpu;
}

TEST_F(AicpuCommunicatorTest, Ut_ClearOpResource_TagNotFound)
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu;
    HcclResult ret = hcclCommAicpu->ClearOpResource("nonexistent_tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete hcclCommAicpu;
}

TEST_F(AicpuCommunicatorTest, Ut_ParseHierarchicalAlgOption_When_SizeExceedMax_Expect_ReturnHCCL_E_INTERNAL)
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu;
    u32 ahcConfInfo[TOP_HIERARCHICAL_CONF_SIZE] = {0};
    ahcConfInfo[TOP_HIERARCHICAL_CONF_lENGTH_INDEX] = TOP_HIERARCHICAL_CONF_SIZE - 1;
    HcclResult ret = hcclCommAicpu->ParseHierarchicalAlgOption(ahcConfInfo);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    delete hcclCommAicpu;
}
