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
#include <securec.h>

// -fno-access-control 已在 CMakeLists.txt 中启用，可直接访问 private 成员
#include "hccl_communicator.h"
#include "coll_alg_param.h"
#include "aicpu_operator_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "launch_aicpu.h"
#include "adapter_rts_common.h"
#include "dispatcher.h"
#include "local_notify.h"
#include "order_launch.h"
#include "profiling_manager_pub.h"
#include "workflow.h"

using namespace hccl;

static int g_launchCallCount = 0;
static HcclResult
LaunchCountStub(const rtStream_t, void*, u32, aclrtBinHandle, const std::string&, bool, u16, void*, u32)
{
    ++g_launchCallCount;
    return HCCL_SUCCESS;
}

class AclgraphCommunicatorTest : public testing::Test {
protected:
    void SetUp() override { GlobalMockObject::reset(); }

    void TearDown() override
    {
        GlobalMockObject::reset();
        // 清理静态成员，避免跨用例污染
        HcclCommunicator::linkResMap_.clear();
        // 清理实例成员，因为 communicator_ 在测试间复用
        communicator_.resMap_.clear();
        communicator_.tagsRequiringHostCleanup_.clear();
        communicator_.rankTagRemoteRes_.clear();
        communicator_.hostMemVec_.clear();
        communicator_.deviceMemVec_.clear();
        communicator_.hostResMap_.clear();
    }

    HcclCommunicator communicator_;
};

/**
 * @brief TC-COMM-01: ClearResMap 正常路径
 * 验证：resMap_ 中存在 tag → findTag=true，tag 被清除
 */
TEST_F(AclgraphCommunicatorTest, ClearResMap_Normal)
{
    std::string tag = "test_tag";
    bool findTag = false;

    communicator_.resMap_[tag] = AlgResourceResponse();

    HcclResult ret = communicator_.ClearResMap(tag, findTag, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(findTag);
    EXPECT_EQ(communicator_.resMap_.count(tag), 0);
}

/**
 * @brief TC-COMM-02: ClearResMap tag 不存在
 * 验证：resMap_ 中不存在 tag → findTag=false
 */
TEST_F(AclgraphCommunicatorTest, ClearResMap_TagNotFound)
{
    std::string tag = "nonexistent_tag";
    bool findTag = false;

    HcclResult ret = communicator_.ClearResMap(tag, findTag, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(findTag);
}

/**
 * @brief TC-COMM-03: ClearResMap 传递 aclGraphDestroyCbk 参数
 * 验证：aclGraphDestroyCbk=true 时正常传递到 DestroyAlgResource
 */
TEST_F(AclgraphCommunicatorTest, ClearResMap_WithDestroyCbk)
{
    std::string tag = "test_tag";
    bool findTag = false;

    communicator_.resMap_[tag] = AlgResourceResponse();

    HcclResult ret = communicator_.ClearResMap(tag, findTag, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(findTag);
    EXPECT_EQ(communicator_.resMap_.count(tag), 0);
}

/**
 * @brief TC-COMM-04: ClearOpResource 清除三种后缀
 * 验证：tag/tag_host/tag_device 三种后缀全部被清除
 */
TEST_F(AclgraphCommunicatorTest, ClearOpResource_AllSuffixes)
{
    std::string tag = "test_tag";

    communicator_.resMap_[tag] = AlgResourceResponse();
    communicator_.resMap_[tag + "_host"] = AlgResourceResponse();
    communicator_.resMap_[tag + "_device"] = AlgResourceResponse();

    HcclResult ret = communicator_.ClearOpResource(tag, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(communicator_.resMap_.count(tag), 0);
    EXPECT_EQ(communicator_.resMap_.count(tag + "_host"), 0);
    EXPECT_EQ(communicator_.resMap_.count(tag + "_device"), 0);
}

/**
 * @brief TC-COMM-05: ClearOpResource 部分后缀存在
 * 验证：只存在 tag/tag_host 时，不崩溃，正常清除存在的条目
 */
TEST_F(AclgraphCommunicatorTest, ClearOpResource_PartialSuffixes)
{
    std::string tag = "test_tag";

    communicator_.resMap_[tag] = AlgResourceResponse();
    communicator_.resMap_[tag + "_host"] = AlgResourceResponse();
    // 缺 _device

    HcclResult ret = communicator_.ClearOpResource(tag, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(communicator_.resMap_.count(tag), 0);
    EXPECT_EQ(communicator_.resMap_.count(tag + "_host"), 0);
    // _device 本来就不存在，不影响
}

/**
 * @brief TC-COMM-06: HasRoceTransportLinks 有 RDMA
 * 验证：transportRequests 中存在 isUsedRdma=true → true
 */
TEST_F(AclgraphCommunicatorTest, HasRoceTransportLinks_WithRdma)
{
    OpCommTransport tpt;
    tpt.resize(1);
    tpt[0].resize(1);
    tpt[0][0].transportRequests.resize(1);
    tpt[0][0].transportRequests[0].isUsedRdma = true;

    EXPECT_TRUE(communicator_.HasRoceTransportLinks(tpt));
}

/**
 * @brief TC-COMM-07: HasRoceTransportLinks 无 RDMA
 * 验证：transportRequests 中 isUsedRdma=false → false
 */
TEST_F(AclgraphCommunicatorTest, HasRoceTransportLinks_NoRdma)
{
    OpCommTransport tpt;
    tpt.resize(1);
    tpt[0].resize(1);
    tpt[0][0].transportRequests.resize(1);
    tpt[0][0].transportRequests[0].isUsedRdma = false;

    EXPECT_FALSE(communicator_.HasRoceTransportLinks(tpt));
}

/**
 * @brief TC-COMM-08: DestroyAlgResource 参数传递验证
 * 验证：aclGraphDestroyCbk=true 时，DestroyOpTransportResponse 中
 * 非 RDMA 链路被保留（linkResMap_ 中不擦除），验证参数正确传递
 */
TEST_F(AclgraphCommunicatorTest, DestroyAlgResource_WithDestroyCbk)
{
    std::string tag = "test_tag";
    bool findTag = false;

    // 预置 2 条链路：非 RDMA。aclGraphDestroyCbk=true 时应被保留
    AlgResourceResponse res;
    res.opTransportResponse.resize(1);
    res.opTransportResponse[0].resize(1);
    res.opTransportResponse[0][0].virtualLinks.resize(1);
    res.opTransportResponse[0][0].virtualLinks[0] = std::make_shared<Transport>();
    res.opTransportResponse[0][0].links.resize(1);
    res.opTransportResponse[0][0].links[0] = std::make_shared<Transport>();
    res.opTransportResponse[0][0].transportRequests.resize(1);
    res.opTransportResponse[0][0].transportRequests[0].isUsedRdma = false; // 非 RDMA
    res.opTransportResponse[0][0].transportRequests[0].isValid = true;

    communicator_.resMap_[tag] = res;
    communicator_.linkResMap_[res.opTransportResponse[0][0].virtualLinks[0].get()] = LinkInfo();
    communicator_.linkResMap_[res.opTransportResponse[0][0].links[0].get()] = LinkInfo();

    // aclGraphDestroyCbk=true 时，非 RDMA 链路保留；RDMA 链路擦除
    HcclResult ret = communicator_.ClearResMap(tag, findTag, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(findTag);
    EXPECT_EQ(communicator_.resMap_.count(tag), 0);
    // 非 RDMA 链路应保留在 linkResMap_ 中
    EXPECT_EQ(communicator_.linkResMap_.size(), 2);
    communicator_.linkResMap_.clear();
}

/**
 * @brief TC-COMM-09: ClearAclgraphHostLinks 简化版
 * 验证：tag 在 tagsRequiringHostCleanup_ 中时，清理后 tag 被移除
 */
TEST_F(AclgraphCommunicatorTest, ClearAclgraphHostLinks_Simplified)
{
    std::string tag = "test_tag";
    communicator_.tagsRequiringHostCleanup_.insert(tag);

    HcclResult ret = communicator_.ClearAclgraphHostLinks({tag});
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(communicator_.tagsRequiringHostCleanup_.count(tag), 0);
}

/**
 * @brief TC-COMM-10: ClearAclgraphHostLinks tag 不在集合中
 * 验证：tag 不在 tagsRequiringHostCleanup_ 中时，无操作，返回 SUCCESS
 */
TEST_F(AclgraphCommunicatorTest, ClearAclgraphHostLinks_TagNotTracked)
{
    // tag 不在 tagsRequiringHostCleanup_ 中，应跳过
    HcclResult ret = communicator_.ClearAclgraphHostLinks({"untracked_tag"});
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * @brief TC-COMM-10b: ClearAclgraphHostLinks 完整路径
 * 验证：rankTagRemoteRes_ 有匹配 tag、hostPtr 非空、hostMemVec_ 含匹配指针时，
 *       ListCommonRemove 被调用，三容器配对 erase
 */
TEST_F(AclgraphCommunicatorTest, ClearAclgraphHostLinks_FullPath)
{
    const std::string tag = "test_tag";
    const u32 rankId = 0;

    // 1. 预置 host 端 HccltagRemoteResV2（模拟通信域分配的 remote res）
    HccltagRemoteResV2 remoteResV2;
    ListCommonInit(&remoteResV2.nextTagRes, &remoteResV2.nextTagRes);
    HccltagRemoteResV2* hostPtr = &remoteResV2;
    // 将 HostMem 包装 remoteResV2 的指针后存入 hostMemVec_
    auto hostMem = std::make_shared<HostMem>(HostMem::create(hostPtr, sizeof(remoteResV2)));
    communicator_.hostMemVec_.push_back(hostMem);
    communicator_.deviceMemVec_.push_back(nullptr);

    // 2. pre-populate tagsRequiringHostCleanup_
    communicator_.tagsRequiringHostCleanup_.insert(tag);

    // 3. pre-populate rankTagRemoteRes_ with HccltagRemoteResV3 pointing to remoteResV2
    HccltagRemoteResV3 remoteResV3;
    remoteResV3.tagRemoteResPtr = hostPtr;
    communicator_.rankTagRemoteRes_[rankId][tag] = remoteResV3;

    // 4. 调用 ClearAclgraphHostLinks
    HcclResult ret = communicator_.ClearAclgraphHostLinks({tag});
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 5. 验证：tagsRequiringHostCleanup_ 已清除
    EXPECT_EQ(communicator_.tagsRequiringHostCleanup_.count(tag), 0);
    // 验证：rankTagRemoteRes_ 中已无此 tag
    EXPECT_EQ(communicator_.rankTagRemoteRes_[rankId].count(tag), 0);
    // 验证：hostMemVec_ 中对应的条目已被 erase
    bool foundInHostMemVec = false;
    for (auto& hm : communicator_.hostMemVec_) {
        if (hm && hm->ptr() == hostPtr) {
            foundInHostMemVec = true;
            break;
        }
    }
    EXPECT_FALSE(foundInHostMemVec);
}

/**
 * @brief TC-COMM-11: AicpuKfcClearOpResLaunch 空 tags
 * 验证：传入空集合时直接返回 SUCCESS
 */
TEST_F(AclgraphCommunicatorTest, KfcClearOpResLaunch_EmptyTags)
{
    HcclResult ret = communicator_.AicpuKfcClearOpResLaunch({});
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * @brief TC-COMM-12: AicpuKfcClearOpResLaunch binHandle_ 为空
 * 验证：binHandle_=nullptr 时跳过 aicpu 清理，返回 SUCCESS
 */
TEST_F(AclgraphCommunicatorTest, KfcClearOpResLaunch_NullBinHandle)
{
    communicator_.binHandle_ = nullptr;
    HcclResult ret = communicator_.AicpuKfcClearOpResLaunch({"tag1"});
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * @brief TC-COMM-13: AicpuKfcClearOpResLaunch opStream_ 为空
 * 验证：opStream_.ptr()=nullptr 时跳过 aicpu 清理，返回 SUCCESS
 */
TEST_F(AclgraphCommunicatorTest, KfcClearOpResLaunch_NullOpStream)
{
    // 预置 binHandle_ 非空但 opStream_ 为空
    communicator_.binHandle_ = reinterpret_cast<aclrtBinHandle>(0x1);
    // opStream_ 默认空
    HcclResult ret = communicator_.AicpuKfcClearOpResLaunch({"tag1"});
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * @brief TC-COMM-14: AicpuKfcClearOpResLaunch launch 路径（单批）
 * 验证：binHandle_ 和 opStream_ 均有效时，进入 launch 路径，
 *       mock hrtMemSyncCopy/AicpuAclKernelLaunchV2/hcclStreamSynchronize 返回成功
 */
TEST_F(AclgraphCommunicatorTest, KfcClearOpResLaunch_LaunchPath)
{
    // 预置成功能进 launch 路径的条件
    communicator_.binHandle_ = reinterpret_cast<aclrtBinHandle>(0x1);
    // 直接设置内部 stream_ 指针，避免 Stream(fake_ptr) 构造函数调用 InitStream 触发 hrt 函数访问
    communicator_.opStream_.stream_ = reinterpret_cast<void*>(0x1234);
    communicator_.identifier_ = "test_group";
    // 预置 buffer 避免走 DeviceMem::alloc 路径（需要 mock 该静态方法）
    communicator_.aicpuCleanupBuf_ = DeviceMem(reinterpret_cast<void*>(0x5678), 4096, false);
    communicator_.aicpuCleanupHostBuf_.reset(new (std::nothrow) HcclKfcClearOpResTilingData());
    ASSERT_NE(communicator_.aicpuCleanupHostBuf_, nullptr);

    // Mock C 函数
    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunchV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = communicator_.AicpuKfcClearOpResLaunch({"tag1", "tag2"});
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * @brief TC-COMM-15: AicpuKfcClearOpResLaunch launch 路径（分批）
 * 验证：tags 数量超过 MAX_BATCH 时自动分批，验证 launch 调用次数
 */
TEST_F(AclgraphCommunicatorTest, KfcClearOpResLaunch_MultiBatch)
{
    communicator_.binHandle_ = reinterpret_cast<aclrtBinHandle>(0x1);
    communicator_.opStream_.stream_ = reinterpret_cast<void*>(0x1234);
    communicator_.identifier_ = "test_group";
    communicator_.aicpuCleanupBuf_ = DeviceMem(reinterpret_cast<void*>(0x5678), 4096, false);
    communicator_.aicpuCleanupHostBuf_.reset(new (std::nothrow) HcclKfcClearOpResTilingData());
    ASSERT_NE(communicator_.aicpuCleanupHostBuf_, nullptr);

    // 构造超过 MAX_BATCH 的 tags
    std::unordered_set<std::string> manyTags;
    for (u32 i = 0; i < HCCL_KFC_CLEAR_OP_RES_MAX_BATCH + 10; i++) {
        manyTags.insert("tag_" + std::to_string(i));
    }

    g_launchCallCount = 0;
    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunchV2).stubs().will(invoke(LaunchCountStub));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = communicator_.AicpuKfcClearOpResLaunch(manyTags);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 应分 2 批 launch（MAX_BATCH + 10 需要 2 批）
    EXPECT_EQ(g_launchCallCount, 2);
}

/**
 * @brief AIT-01: AicpuInitOpTilingDataAicpuCache enable=0
 * 验证：aicpuCacheEnable=0 时，不修改 tilingData 值
 */
TEST_F(AclgraphCommunicatorTest, AicpuInitTilingCache_Disabled)
{
    OpParam opParam;
    opParam.aicpuCacheEnable = 0;
    OpTilingData tilingData;
    tilingData.aicpuCacheEnable = 0;

    HcclResult ret = communicator_.AicpuInitOpTilingDataAicpuCache(opParam, HCCL_CMD_ALLREDUCE, &tilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tilingData.aicpuCacheEnable, 0);
}

/**
 * @brief AIT-02: AicpuInitOpTilingDataAicpuCache enable=1 + isCapture
 * 验证：算子模式 isCapture=true 情况下能正常返回
 */
TEST_F(AclgraphCommunicatorTest, AicpuInitTilingCache_ForceOpbaseWithCapture)
{
    OpParam opParam;
    opParam.aicpuCacheEnable = 1;
    opParam.isCapture = true;
    opParam.isZeroCopy = true;
    OpTilingData tilingData;
    tilingData.aicpuCacheEnable = 0;

    HcclResult ret = communicator_.AicpuInitOpTilingDataAicpuCache(opParam, HCCL_CMD_ALLREDUCE, &tilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * @brief AIT-03: AicpuInitOpTilingDataAicpuCache enable >= 10
 * 验证：aicpuCacheEnable >= 10 时触发内部错误检查
 */
TEST_F(AclgraphCommunicatorTest, AicpuInitTilingCache_EnableTooHigh)
{
    // 默认构造的 communicator 不在图模式下，外层 if 条件不满足，
    // aicpuCacheEnable >= 10 的校验分支不会触发，返回 SUCCESS
    OpParam opParam;
    opParam.aicpuCacheEnable = 10;
    OpTilingData tilingData;

    HcclResult ret = communicator_.AicpuInitOpTilingDataAicpuCache(opParam, HCCL_CMD_ALLREDUCE, &tilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tilingData.aicpuCacheEnable, 10U);
}

/**
 * @brief AIT-04: AicpuInitOpTilingDataAicpuCache 单算子模式
 * 验证：非图模式下，直接赋值 aicpuCacheEnable 不变
 */
TEST_F(AclgraphCommunicatorTest, AicpuInitTilingCache_OpsMode)
{
    OpParam opParam;
    opParam.aicpuCacheEnable = 1;
    OpTilingData tilingData;
    tilingData.aicpuCacheEnable = 0;

    HcclResult ret = communicator_.AicpuInitOpTilingDataAicpuCache(opParam, HCCL_CMD_ALLREDUCE, &tilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tilingData.aicpuCacheEnable, 1);
}

/**
 * @brief AIL-01: AicpuKfcTilingDataLaunchIn capture 成功路径
 * 验证：isCapture=true 时走完两段 event 按序同步逻辑，返回 SUCCESS，
 *       覆盖 KERNEL_INDEX_ACLGRAPH_EVENT_0/1 的 record/wait 及按序下发新增代码行。
 */
TEST_F(AclgraphCommunicatorTest, AicpuKfcTilingDataLaunchIn_CaptureSuccess)
{
    OpParam opParam;
    opParam.tag = "test_tag";
    opParam.stream.stream_ = reinterpret_cast<void*>(0x1000);
    opParam.inputPtr = nullptr;
    opParam.outputPtr = nullptr;
    opParam.isCapture = true;
    opParam.opType = HCCL_CMD_ALLREDUCE;

    DeviceMem deviceContext;
    std::string kernelName = "test_kernel";
    AicpuOpTiling opTilingInfo{};

    communicator_.deviceLogicId_ = 0;
    communicator_.identifier_ = "test_group";
    communicator_.opStream_.stream_ = reinterpret_cast<void*>(0x2000);

    MOCKER(SetNormalMode).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LocalNotify::Post).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LocalNotify::Wait).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    MOCKER(hrtMsprofSysCycleTime).stubs().will(returnValue(static_cast<uint64_t>(0)));
    MOCKER(aclrtRecordEvent).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtStreamWaitEvent).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(&OrderLaunch::AclgraphLaunchInOrderToOrderStream).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OrderLaunch::AclgraphLaunchInOrderToKernelStream).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::KernelLaunchChooseAicpuOrCustom).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(ProfilingManagerPub::CallMsprofReportNodeInfo).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret
        = communicator_.AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * @brief AIL-02: AicpuKfcTilingDataLaunchIn capture 首次 recordEvent 失败
 * 验证：第一段 aclrtRecordEvent 返回非 ACL_SUCCESS 时，CHK_PRT_RET 提前返回 HCCL_E_RUNTIME。
 */
TEST_F(AclgraphCommunicatorTest, AicpuKfcTilingDataLaunchIn_CaptureRecordEventFail)
{
    OpParam opParam;
    opParam.tag = "test_tag";
    opParam.stream.stream_ = reinterpret_cast<void*>(0x1000);
    opParam.isCapture = true;
    opParam.opType = HCCL_CMD_ALLREDUCE;

    DeviceMem deviceContext;
    std::string kernelName = "test_kernel";
    AicpuOpTiling opTilingInfo{};

    communicator_.deviceLogicId_ = 0;
    communicator_.identifier_ = "test_group";
    communicator_.opStream_.stream_ = reinterpret_cast<void*>(0x2000);

    MOCKER(SetNormalMode).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LocalNotify::Post).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    MOCKER(hrtMsprofSysCycleTime).stubs().will(returnValue(static_cast<uint64_t>(0)));
    MOCKER(aclrtRecordEvent).stubs().will(returnValue(1));

    HcclResult ret
        = communicator_.AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo, 0, false);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}
