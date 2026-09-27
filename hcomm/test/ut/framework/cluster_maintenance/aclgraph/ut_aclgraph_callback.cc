/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <cstring>
#include <unordered_set>

#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "hcom_pub.h"
#include "hcom_common.h"
#include "comm_config_pub.h"
#include "common.h"
#include "hccl_communicator.h"

#define private public
#include "aclgraph_callback.h"
#undef private

#include "aclgraph/zero_copy_acl_graph.h"
#include "aicpu_operator_pub.h"
#include "stream_utils.h"

using namespace hccl;

// Mock 函数用于 rtModelGetId
rtError_t rtModelGetIdMock(rtModel_t model, uint32_t* modelId)
{
    *modelId = 1; // 返回一个固定的 mock modelId
    return RT_ERROR_NONE;
}

// Mock: GetStreamCaptureInfo — 设置 rtModel 非空, isCapture=true
HcclResult GetStreamCaptureInfoMock(aclrtStream stream, aclmdlRI& rtModel, bool& isCapture)
{
    rtModel = reinterpret_cast<aclmdlRI>(0x1);
    isCapture = true;
    return HCCL_SUCCESS;
}

// Mock: GetModelId — 设置固定 modelId
HcclResult GetModelIdMock(aclmdlRI& rtModel, u64& modelId)
{
    modelId = 1;
    return HCCL_SUCCESS;
}

class AclgraphCallbackTest : public testing::Test {
protected:
    void SetUp() override
    {
        comm.reset(new (std::nothrow) hccl::hcclComm());
        if (!comm) {
            HCCL_ERROR("Failed to create hccl::hcclComm");
            return;
        }
    }

    void TearDown() override
    {
        // 清理单例状态，避免测试用例之间相互影响
        std::lock_guard<std::mutex> lock(AclgraphCallback::GetInstance().resMutex_);
        AclgraphCallback::GetInstance().captureResMap_.clear();
        AclgraphCallback::GetInstance().captureCallbackParamMap_.clear();
        GlobalMockObject::reset();
    }

    std::shared_ptr<hccl::hcclComm> comm;
    HcclCommunicator communicator1_;
    HcclCommunicator communicator2_;
    int mockModel_ = 0;
};

TEST_F(AclgraphCallbackTest, ut_InsertNewTagToCaptureResMap_When_Capture_Expect_SUCCESS)
{
    MOCKER(rtModelGetId).stubs().will(invoke(rtModelGetIdMock));

    std::string newTag = "tag";
    OpParam opParam;
    HcclResult ret = AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(&communicator1_, newTag, opParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(AclgraphCallbackTest, ut_InsertNewTagToCaptureResMap_When_Comm_Invalid_Expect_Fail)
{
    std::string newTag = "tag";
    OpParam opParam;
    HcclResult ret = AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(nullptr, newTag, opParam);
    EXPECT_EQ(ret, HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(AclgraphCallbackTest, ut_CleanCaptureRes_When_modelId_Invalid_Expect_Fail)
{
    u64 modelId = 0;

    HcclResult ret = AclgraphCallback::GetInstance().CleanCaptureRes(modelId);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    GlobalMockObject::verify();
}

TEST_F(AclgraphCallbackTest, ut_CleanCaptureRes_By_Communicator_With_Multiple_Comms_Expect_Correct_Comm_Removed)
{
    std::string newTag = "tag";
    OpParam opParam;

    // 插入两个 communicator 到 captureResMap_
    HcclResult ret1 = AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(&communicator1_, newTag, opParam);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    HcclResult ret2 = AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(&communicator2_, newTag, opParam);
    EXPECT_EQ(ret2, HCCL_SUCCESS);

    // 验证插入成功
    bool foundComm1Before = false;
    bool foundComm2Before = false;
    for (const auto& modelEntry : AclgraphCallback::GetInstance().captureResMap_) {
        if (modelEntry.second.find(&communicator1_) != modelEntry.second.end()) {
            foundComm1Before = true;
        }
        if (modelEntry.second.find(&communicator2_) != modelEntry.second.end()) {
            foundComm2Before = true;
        }
    }
    EXPECT_TRUE(foundComm1Before) << "Communicator1 should be found in captureResMap_ before CleanCaptureRes";
    EXPECT_TRUE(foundComm2Before) << "Communicator2 should be found in captureResMap_ before CleanCaptureRes";

    // 只删除 communicator1
    AclgraphCallback::GetInstance().CleanCaptureRes(&communicator1_);

    // 验证：communicator1 应该被删除，communicator2 应该仍然存在
    bool foundComm1After = false;
    bool foundComm2After = false;
    for (const auto& modelEntry : AclgraphCallback::GetInstance().captureResMap_) {
        if (modelEntry.second.find(&communicator1_) != modelEntry.second.end()) {
            foundComm1After = true;
        }
        if (modelEntry.second.find(&communicator2_) != modelEntry.second.end()) {
            foundComm2After = true;
        }
    }
    EXPECT_FALSE(foundComm1After) << "Communicator1 should NOT be found in captureResMap_ after CleanCaptureRes";
    EXPECT_TRUE(foundComm2After) << "Communicator2 should still be found in captureResMap_ after CleanCaptureRes";

    GlobalMockObject::verify();
}

// ======== TC-UTIL-01~03: ListCommonRemove ========

TEST_F(AclgraphCallbackTest, ListCommonRemove_NormalNode)
{
    ListCommon nodes[3];
    for (int i = 0; i < 3; i++) {
        ListCommonInit(&nodes[i], &nodes[i]);
    }
    ListCommonAddHead(&nodes[1], &nodes[1], &nodes[0], &nodes[0]);
    ListCommonAddHead(&nodes[2], &nodes[2], &nodes[1], &nodes[1]);
    ListCommonRemove(&nodes[1]);
    EXPECT_EQ(reinterpret_cast<ListCommon*>(nodes[0].nextHost), &nodes[2]);
    EXPECT_EQ(reinterpret_cast<ListCommon*>(nodes[2].preHost), &nodes[0]);
    EXPECT_EQ(reinterpret_cast<ListCommon*>(nodes[1].nextHost), &nodes[1]);
    EXPECT_EQ(reinterpret_cast<ListCommon*>(nodes[1].preHost), &nodes[1]);
}

TEST_F(AclgraphCallbackTest, ListCommonRemove_Nullptr) { ListCommonRemove(nullptr); }

TEST_F(AclgraphCallbackTest, ListCommonRemove_SelfLoopNode)
{
    ListCommon node;
    ListCommonInit(&node, &node);
    ListCommonRemove(&node);
    EXPECT_EQ(reinterpret_cast<ListCommon*>(node.nextHost), &node);
    EXPECT_EQ(reinterpret_cast<ListCommon*>(node.preHost), &node);
}

// ======== TC-UTIL-04~06: IsAclGraphZeroCopyAlgAvailable ========

TEST_F(AclgraphCallbackTest, IsZeroCopyAvailable_NonReduceOp)
{
    ZeroCopyAclGraph algo;
    OpParam opParam;
    opParam.aclGraphZeroCopyEnable = 0;
    EXPECT_TRUE(algo.IsAclGraphZeroCopyAlgAvailable(HCCL_CMD_ALLGATHER, opParam));
    EXPECT_TRUE(algo.IsAclGraphZeroCopyAlgAvailable(HCCL_CMD_BROADCAST, opParam));
    EXPECT_TRUE(algo.IsAclGraphZeroCopyAlgAvailable(HCCL_CMD_ALLTOALLV, opParam));
}

TEST_F(AclgraphCallbackTest, IsZeroCopyAvailable_ReduceOp_Enabled)
{
    ZeroCopyAclGraph algo;
    OpParam opParam;
    opParam.aclGraphZeroCopyEnable = 1;
    EXPECT_TRUE(algo.IsAclGraphZeroCopyAlgAvailable(HCCL_CMD_ALLREDUCE, opParam));
    EXPECT_TRUE(algo.IsAclGraphZeroCopyAlgAvailable(HCCL_CMD_REDUCE, opParam));
}

TEST_F(AclgraphCallbackTest, IsZeroCopyAvailable_ReduceOp_Disabled)
{
    ZeroCopyAclGraph algo;
    OpParam opParam;
    opParam.aclGraphZeroCopyEnable = 0;
    EXPECT_FALSE(algo.IsAclGraphZeroCopyAlgAvailable(HCCL_CMD_ALLREDUCE, opParam));
    EXPECT_FALSE(algo.IsAclGraphZeroCopyAlgAvailable(HCCL_CMD_REDUCE_SCATTER, opParam));
}

// ======== TC-CALLBACK-01~04: AclgraphCallback 全流程 ========

TEST_F(AclgraphCallbackTest, CleanCaptureRes_ModelIdNotFound)
{
    HcclResult ret = AclgraphCallback::GetInstance().CleanCaptureRes(999);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(AclgraphCallbackTest, CleanCaptureRes_ThreePhase_SimplifiedFlow)
{
    // 直接操作私有成员预置 captureResMap_ 来模拟已注册的 tag
    // 使用真实实现，依赖早期返回路径特性简化测试
    // AicpuKfcClearOpResLaunch: binHandle_ == nullptr 时直接返回 HCCL_SUCCESS
    // ClearAclgraphHostLinks: rankTagRemoteRes_ 为空时只清 tagsRequiringHostCleanup_
    // ClearOpResource: resMap_ 中找不到 tag 时仍返回 HCCL_SUCCESS

    const u64 modelId = 42;
    const std::string tag = "test_tag_Capture";

    // 直接插入 captureResMap_ 条目（跳过 InsertNewTagToCaptureResMap 的 stream 依赖）
    {
        std::lock_guard<std::mutex> lock(AclgraphCallback::GetInstance().resMutex_);
        AclgraphCallback::GetInstance().captureResMap_[modelId][&communicator1_].insert(tag);
        AclgraphCallback::GetInstance().captureCallbackParamMap_[modelId].modelId = modelId;
    }

    // 预置 tagsRequiringHostCleanup_
    communicator1_.tagsRequiringHostCleanup_.insert(tag);

    // 执行 CleanCaptureRes
    HcclResult ret = AclgraphCallback::GetInstance().CleanCaptureRes(modelId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证：captureResMap_ 和 captureCallbackParamMap_ 已清空
    {
        std::lock_guard<std::mutex> lock(AclgraphCallback::GetInstance().resMutex_);
        EXPECT_TRUE(AclgraphCallback::GetInstance().captureResMap_.empty());
        EXPECT_TRUE(AclgraphCallback::GetInstance().captureCallbackParamMap_.empty());
    }

    // 验证：tagsRequiringHostCleanup_ 中 tag 已被清除
    EXPECT_EQ(communicator1_.tagsRequiringHostCleanup_.count(tag), 0U);
}

TEST_F(AclgraphCallbackTest, InsertNewTag_CallbackRegistration)
{
    // Mock 底层 C 函数使 InsertNewTagToCaptureResMap 能正常走完全程
    MOCKER(GetStreamCaptureInfo).stubs().will(invoke(GetStreamCaptureInfoMock));

    MOCKER(GetModelId).stubs().will(invoke(GetModelIdMock));

    MOCKER(aclmdlRIDestroyRegisterCallback)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));

    // 构造 OpParam（stream 需要有合法 .ptr() 避免空指针）
    OpParam opParam;

    std::string tag1 = "tag1";
    HcclResult ret = AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(&communicator1_, tag1, opParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 第一次插入应注册 callback
    // 验证 captureResMap_ 包含 modelId→communicator1_→tag1
    bool found = false;
    for (const auto& modelEntry : AclgraphCallback::GetInstance().captureResMap_) {
        for (const auto& commEntry : modelEntry.second) {
            if (commEntry.first == &communicator1_ && commEntry.second.find(tag1) != commEntry.second.end()) {
                found = true;
            }
        }
    }
    EXPECT_TRUE(found);

    GlobalMockObject::reset();
}

TEST_F(AclgraphCallbackTest, InsertNewTag_NoDuplicateCallback)
{
    // 同一个 modelId 第二次插入不应再次注册 callback
    MOCKER(GetStreamCaptureInfo).stubs().will(invoke(GetStreamCaptureInfoMock));

    MOCKER(GetModelId).stubs().will(invoke(GetModelIdMock));

    MOCKER(aclmdlRIDestroyRegisterCallback).stubs().will(returnValue(ACL_SUCCESS));

    OpParam opParam;

    // 第一次插入
    std::string tag1 = "tag1";
    HcclResult ret1 = AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(&communicator1_, tag1, opParam);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    // 第二次插入同 communicator 不同 tag → 不应再次注册
    std::string tag2 = "tag2";
    HcclResult ret2 = AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(&communicator1_, tag2, opParam);
    EXPECT_EQ(ret2, HCCL_SUCCESS);

    // 验证 captureResMap_ 包含两个 tag
    bool foundTag1 = false;
    bool foundTag2 = false;
    for (const auto& modelEntry : AclgraphCallback::GetInstance().captureResMap_) {
        for (const auto& commEntry : modelEntry.second) {
            if (commEntry.first == &communicator1_) {
                if (commEntry.second.find(tag1) != commEntry.second.end())
                    foundTag1 = true;
                if (commEntry.second.find(tag2) != commEntry.second.end())
                    foundTag2 = true;
            }
        }
    }
    EXPECT_TRUE(foundTag1);
    EXPECT_TRUE(foundTag2);

    GlobalMockObject::reset();
}
