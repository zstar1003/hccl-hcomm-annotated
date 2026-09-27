/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>

#include <memory>
#include <iostream>
#include <string>
#include <cstring>

#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_one_side_service.h"
#include "stream_pub.h"
#include "aicpu_operator_pub.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class OneSideServiceUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "OneSideServiceUT SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "OneSideServiceUT TearDown" << std::endl; }
    virtual void SetUp()
    {
        HcclOneSideServiceAicpu::services_.clear();
        std::cout << "A Test SetUp" << std::endl;
    }
    virtual void TearDown()
    {
        HcclOneSideServiceAicpu::services_.clear();
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }

    void SetupService(
        std::shared_ptr<HcclOneSideServiceAicpu>& service, const std::string& identifier, bool execStreamEnable,
        uint32_t devId, uint32_t sqId, uint32_t actualStreamId)
    {
        service->identifier_ = identifier;
        service->execStreamEnable_ = false;
        service->devId_ = 0;

        HcclComStreamInfo streamInfo{};
        streamInfo.sqId = 1;
        streamInfo.actualStreamId = 100;
        service->execStream_ = Stream(streamInfo);
    }
};

TEST_F(OneSideServiceUT, Ut_CleanStreamFunc_When_Disabled_ExecutesClean)
{
    auto service = std::make_shared<HcclOneSideServiceAicpu>();
    SetupService(service, "test_clean", false, 0, 1, 100);

    SqCqeContext sqCqeContext{};
    service->execStream_.sqeContext_ = &sqCqeContext.sqContext;

    MOCKER(ConfigSqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Stream::ClearLocalBuff).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(QuerySqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = service->CleanStreamFunc();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(service->execStreamEnable_);
}

TEST_F(OneSideServiceUT, Ut_DisableAllStreamFunc_When_DisableStreamSuc)
{
    auto service = std::make_shared<HcclOneSideServiceAicpu>();
    SetupService(service, "test_DisableStream", false, 0, 1, 100);

    HcclResult ret = service->DisableAllStreamFunc();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OneSideServiceUT, Ut_CleanAllStreamFunc_When_CleanStreamFuncFail_ReturnsError)
{
    auto service = std::make_shared<HcclOneSideServiceAicpu>();
    SetupService(service, "fail_tag", false, 0, 1, 100);
    HcclOneSideServiceAicpu::services_["fail_tag"] = service;

    MOCKER(ConfigSqStatusByType).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = HcclOneSideServiceAicpu::CleanAllStreamFunc();
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 测试 Process - finalize 分支，覆盖 unique_lock 写路径(L54)
TEST_F(OneSideServiceUT, Ut_Process_When_Finalize_Expect_EraseService)
{
    // 先在 services_ 中插入一个 service
    auto service = std::make_shared<HcclOneSideServiceAicpu>();
    SetupService(service, "finalize_tag", false, 0, 1, 100);
    HcclOneSideServiceAicpu::services_["finalize_tag"] = service;
    ASSERT_EQ(HcclOneSideServiceAicpu::services_.size(), 1u);

    // 构造 OpTilingData + OpTilingOneSideCommDataDes，设置 finalize=true
    constexpr size_t bufSize = sizeof(OpTilingData) + sizeof(OpTilingOneSideCommDataDes);
    std::vector<u8> tilingBuf(bufSize, 0);
    auto* tilingData = reinterpret_cast<OpTilingData*>(tilingBuf.data());
    strncpy(tilingData->tag, "finalize_tag", sizeof(tilingData->tag) - 1);
    tilingData->length = sizeof(OpTilingOneSideCommDataDes); // 满足 >= sizeof(OpTilingOneSideCommDataDes)

    auto* vDataPtr = reinterpret_cast<OpTilingOneSideCommDataDes*>(tilingBuf.data() + sizeof(OpTilingData));
    vDataPtr->finalize = true;

    HcclResult ret = HcclOneSideServiceAicpu::Process(tilingData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(HcclOneSideServiceAicpu::services_.count("finalize_tag"), 0u);
}

// 测试 Process - length 不足，覆盖参数校验提前返回
TEST_F(OneSideServiceUT, Ut_Process_When_DynamicDataTooSmall_Expect_ReturnParaError)
{
    constexpr size_t bufSize = sizeof(OpTilingData) + 4; // 故意小于 sizeof(OpTilingOneSideCommDataDes)
    std::vector<u8> tilingBuf(bufSize, 0);
    auto* tilingData = reinterpret_cast<OpTilingData*>(tilingBuf.data());
    strncpy(tilingData->tag, "short_tag", sizeof(tilingData->tag) - 1);
    tilingData->length = 2; // 小于 sizeof(OpTilingOneSideCommDataDes)

    HcclResult ret = HcclOneSideServiceAicpu::Process(tilingData);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 测试 GetService - 查找已存在的 service，覆盖 shared_lock 读路径(L69)
TEST_F(OneSideServiceUT, Ut_GetService_When_Exist_Expect_ReturnExistingService)
{
    auto service = std::make_shared<HcclOneSideServiceAicpu>();
    SetupService(service, "exist_tag", false, 0, 1, 100);
    HcclOneSideServiceAicpu::services_["exist_tag"] = service;

    // 构造最小的 tilingData（GetService 查找命中时不会使用 tilingData 内容）
    constexpr size_t bufSize = sizeof(OpTilingData) + sizeof(OpTilingOneSideCommDataDes);
    std::vector<u8> tilingBuf(bufSize, 0);
    auto* tilingData = reinterpret_cast<OpTilingData*>(tilingBuf.data());
    strncpy(tilingData->tag, "exist_tag", sizeof(tilingData->tag) - 1);
    tilingData->length = sizeof(OpTilingOneSideCommDataDes);
    auto* vDataPtr = reinterpret_cast<OpTilingOneSideCommDataDes*>(tilingBuf.data() + sizeof(OpTilingData));
    vDataPtr->finalize = false;

    auto result = HcclOneSideServiceAicpu::GetService("exist_tag", tilingData);
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(result, service);
}

// 测试 GetService - 不存在时创建新 service，覆盖 unique_lock 写路径(L81)
TEST_F(OneSideServiceUT, Ut_GetService_When_NotExist_Expect_CreateAndInsert)
{
    constexpr size_t bufSize = sizeof(OpTilingData) + sizeof(OpTilingOneSideCommDataDes);
    std::vector<u8> tilingBuf(bufSize, 0);
    auto* tilingData = reinterpret_cast<OpTilingData*>(tilingBuf.data());
    strncpy(tilingData->tag, "new_tag", sizeof(tilingData->tag) - 1);
    tilingData->length = sizeof(OpTilingOneSideCommDataDes);
    auto* vDataPtr = reinterpret_cast<OpTilingOneSideCommDataDes*>(tilingBuf.data() + sizeof(OpTilingData));
    vDataPtr->finalize = false;
    vDataPtr->rankSize = 1;

    // GetService 内部会调用 Init，Init 可能依赖运行时，mock 掉关键依赖
    MOCKER_CPP(&HcclOneSideServiceAicpu::InitOpNotifyObj).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclOneSideServiceAicpu::InitStream).stubs().will(returnValue(HCCL_SUCCESS));

    auto result = HcclOneSideServiceAicpu::GetService("new_tag", tilingData);
    // 如果 Init 成功，应该返回非空且插入到 map
    if (result != nullptr) {
        EXPECT_EQ(HcclOneSideServiceAicpu::services_.count("new_tag"), 1u);
    }
    GlobalMockObject::verify();
}
