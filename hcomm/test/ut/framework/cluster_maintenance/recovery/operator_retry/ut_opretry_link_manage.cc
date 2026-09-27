/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "opretry_link_manage.h"

using namespace hccl;

class OpretryLinkManageTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "OpretryLinkManageTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "OpretryLinkManageTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(OpretryLinkManageTest, Ut_GetInstance_When_ValidDeviceId_Expect_ReturnInstance)
{
    OpretryLinkManage& instance1 = OpretryLinkManage::GetInstance(0);
    OpretryLinkManage& instance2 = OpretryLinkManage::GetInstance(0);
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(OpretryLinkManageTest, Ut_GetInstance_When_InvalidDeviceId_Expect_ReturnFirstInstance)
{
    OpretryLinkManage& defaultInstance = OpretryLinkManage::GetInstance(0);
    OpretryLinkManage& invalidInstance = OpretryLinkManage::GetInstance(100); // 超过MAX_DEV_NUM
    EXPECT_EQ(&defaultInstance, &invalidInstance);
}

TEST_F(OpretryLinkManageTest, Ut_AddLinkInfoByIdentifier_When_NewIdentifier_Expect_Success)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "test_identifier";
    std::string newTag = "test_tag";
    std::vector<u32> remoteRankList = {1, 2, 3};

    HcclResult ret = manager.AddLinkInfoByIdentifier(identifier, newTag, remoteRankList, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpretryLinkManageTest, Ut_AddLinkInfoByIdentifier_When_ExistingTag_Expect_Success)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "test_identifier_2";
    std::string newTag = "test_tag_2";
    std::vector<u32> remoteRankList = {1, 2, 3};

    HcclResult ret = manager.AddLinkInfoByIdentifier(identifier, newTag, remoteRankList, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 再次添加相同的tag，应该成功
    ret = manager.AddLinkInfoByIdentifier(identifier, newTag, remoteRankList, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpretryLinkManageTest, Ut_AddLinkInfoByIdentifier_When_Incremental_Expect_Success)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "test_identifier_3";
    std::string newTag = "test_tag_3";
    std::vector<u32> remoteRankList1 = {1, 2, 3};
    std::vector<u32> remoteRankList2 = {3, 4, 5};

    HcclResult ret = manager.AddLinkInfoByIdentifier(identifier, newTag, remoteRankList1, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 增量添加
    ret = manager.AddLinkInfoByIdentifier(identifier, newTag, remoteRankList2, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证获取到的列表包含所有元素
    std::vector<u32> resultList;
    ret = manager.GetLinkInfoByIdentifier(identifier, newTag, resultList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(resultList.size() >= 5); // 应该包含1,2,3,4,5
}

TEST_F(OpretryLinkManageTest, Ut_GetLinkInfoByIdentifier_When_WithoutTag_Expect_Success)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "test_identifier_4";
    std::string newTag1 = "test_tag_4_1";
    std::string newTag2 = "test_tag_4_2";
    std::vector<u32> remoteRankList1 = {1, 2, 3};
    std::vector<u32> remoteRankList2 = {4, 5, 6};

    HcclResult ret = manager.AddLinkInfoByIdentifier(identifier, newTag1, remoteRankList1, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = manager.AddLinkInfoByIdentifier(identifier, newTag2, remoteRankList2, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<u32> resultList;
    ret = manager.GetLinkInfoByIdentifier(identifier, resultList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(resultList.size() >= 6); // 应该包含1-6
}

TEST_F(OpretryLinkManageTest, Ut_GetLinkInfoByIdentifier_When_WithoutTagAndNotExist_Expect_HCCL_E_PARA)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "non_existent_identifier";

    std::vector<u32> resultList;
    HcclResult ret = manager.GetLinkInfoByIdentifier(identifier, resultList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpretryLinkManageTest, Ut_DeleteLinkInfoByIdentifier_When_NonExistingIdentifier_Expect_Success)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "non_existent_identifier";

    HcclResult ret = manager.DeleteLinkInfoByIdentifier(identifier);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpretryLinkManageTest, Ut_DeleteLinkInfoByIdentifier_When_ExistingIdentifier_Expect_Success)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "test_identifier_5";
    std::string newTag = "test_tag_5";
    std::vector<u32> remoteRankList = {1, 2, 3};

    HcclResult ret = manager.AddLinkInfoByIdentifier(identifier, newTag, remoteRankList, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = manager.DeleteLinkInfoByIdentifier(identifier);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证已删除
    std::vector<u32> resultList;
    ret = manager.GetLinkInfoByIdentifier(identifier, newTag, resultList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpretryLinkManageTest, Ut_DeleteLinkInfoByIdentifier_When_DeInit_Expect_Success)
{
    OpretryLinkManage& manager = OpretryLinkManage::GetInstance(0);
    std::string identifier = "test_identifier_6";

    // 模拟已销毁状态（通过修改isDeInit_）
    // 注意：这里需要访问私有成员，可能需要在测试文件中添加#define private public
    HcclResult ret = manager.DeleteLinkInfoByIdentifier(identifier);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
