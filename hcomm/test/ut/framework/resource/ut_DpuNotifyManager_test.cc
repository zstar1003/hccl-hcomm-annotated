/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "dpu_notify_manager.h"

#include <string>

using namespace hccl;

class DpuNotifyManagerTest : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    std::string VectorToStr(const std::vector<uint32_t>& v)
    {
        std::string result;

        constexpr int maxShow = 32;

        for (int i = 0; i < v.size(); ++i) {
            result += std::to_string(v[i]);
            if (i >= maxShow) {
                result += "...";
                break;
            }
            if (i < v.size() - 1) {
                result += ", ";
            }
        }

        return result;
    }
};

/*
    Allocate 9 notifies, then free 9 notifies.
 */
TEST_F(DpuNotifyManagerTest, ut_DpuNotifyManager_When_Alloc1Free1_Expect_ReturnIsHCCL_SUCCESS)
{
    HcclResult ret = HCCL_E_RESERVED;

    std::vector<uint32_t> notifyIds;
    const uint32_t notifyNum = 9;
    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum, notifyIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(VectorToStr(notifyIds), "0, 1, 2, 3, 4, 5, 6, 7, 8");

    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum, notifyIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/*
    Allocate 5 notifies for list1,
    then allocate 4 notifies for list2.
    Finally, free the two lists.
*/
TEST_F(DpuNotifyManagerTest, ut_DpuNotifyManager_When_Alloc2Free2_Expect_ReturnIsHCCL_SUCCESS)
{
    HcclResult ret = HCCL_E_RESERVED;

    std::vector<uint32_t> notifyIds1;
    std::vector<uint32_t> notifyIds2;
    const uint32_t notifyNum1 = 5;
    const uint32_t notifyNum2 = 4;
    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum1, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum2, notifyIds2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(VectorToStr(notifyIds1), "0, 1, 2, 3, 4");
    EXPECT_EQ(VectorToStr(notifyIds2), "5, 6, 7, 8");

    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum1, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum2, notifyIds2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/*
    Allocate 4 notifies for list1,          (11110000 00000000)
    then allocate 5 notifies for list2.     (11111111 10000000)
    Free the list1,                         (00001111 10000000)
    then allocate 5 notifies for list1.     (11111111 11000000)
    Finally, free the two lists.            (00000000 00000000)
*/
TEST_F(DpuNotifyManagerTest, ut_DpuNotifyManager_When_Alloc2Free1AllocBigger1_Expect_ReturnIsHCCL_SUCCESS)
{
    HcclResult ret = HCCL_E_RESERVED;

    std::vector<uint32_t> notifyIds1;
    std::vector<uint32_t> notifyIds2;
    const uint32_t notifyNum1 = 4;
    const uint32_t notifyNum2 = 5;
    const uint32_t notifyNum1Bigger = 5;
    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum1, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum2, notifyIds2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(VectorToStr(notifyIds1), "0, 1, 2, 3");
    EXPECT_EQ(VectorToStr(notifyIds2), "4, 5, 6, 7, 8");

    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum1, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum1Bigger, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(VectorToStr(notifyIds1), "0, 1, 2, 3, 9");

    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum1Bigger, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum2, notifyIds2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DpuNotifyManagerTest, ut_DpuNotifyManager_When_notifyNumIsZero_Expect_ReturnIsHCCL_SUCCESS)
{
    HcclResult ret = HCCL_E_RESERVED;

    std::vector<uint32_t> notifyIds;
    const uint32_t notifyNum = 0;

    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum, notifyIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum, notifyIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DpuNotifyManagerTest, ut_DpuNotifyManager_When_notifyNumTooLarge_Expect_ReturnIsHCCL_E_PARA)
{
    HcclResult ret = HCCL_E_RESERVED;

    std::vector<uint32_t> notifyIds;
    const uint32_t notifyNum = 8193;

    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum, notifyIds);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(DpuNotifyManagerTest, ut_DpuNotifyManager_When_secondAllocOverflow_Expect_ReturnIsHCCL_E_MEMORY)
{
    HcclResult ret = HCCL_E_RESERVED;

    std::vector<uint32_t> notifyIds1;
    std::vector<uint32_t> notifyIds2;
    const uint32_t notifyNum1 = 8183;
    const uint32_t notifyNum2 = 8193 - notifyNum1;
    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum1, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum2, notifyIds2);
    EXPECT_EQ(ret, HCCL_E_MEMORY);

    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum1, notifyIds1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DpuNotifyManagerTest, ut_DpuNotifyManager_When_freeWithInvalidNum_Expect_ReturnIsHCCL_E_PARA)
{
    HcclResult ret = HCCL_E_RESERVED;

    std::vector<uint32_t> notifyIds;
    const uint32_t notifyNum = 4;
    const uint32_t notifyNumInvalid = 5;

    ret = DpuNotifyManager::GetInstance().AllocNotifyIds(notifyNum, notifyIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNumInvalid, notifyIds);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = DpuNotifyManager::GetInstance().FreeNotifyIds(notifyNum, notifyIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
