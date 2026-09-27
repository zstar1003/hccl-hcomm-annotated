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
#include "hccl_tbe_task.h"
#include "../../hccl_api_base_test.h"
#include "hccl/hccl_res.h"

static const char* RANKTABLE_FILE_NAME = nullptr;
static constexpr uint64_t MB_UNIT = 1 * 1024 * 1024;
class HcclIndependentOpMemTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        MOCKER(HcclTbeTaskInit).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(&HcclCommunicator::InitRaResource).stubs().will(returnValue(HCCL_SUCCESS));
        UT_USE_RANK_TABLE_910_1SERVER_2RANK;
        RANKTABLE_FILE_NAME = rankTableFileName;
        EXPECT_EQ(RANKTABLE_FILE_NAME != nullptr, true);
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclIndependentOpMemTest, Ut_HcclGetHcclBuffer_When_Param_Is_Invalid_Expect_Para_Error)
{
    UT_COMM_CREATE_DEFAULT(comm);
    void* buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = HcclGetHcclBuffer(comm, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclGetHcclBuffer(nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclGetHcclBuffer(comm, &buffer, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclIndependentOpMemTest, Ut_HcclGetHcclBuffer_When_Get_Default_Mem_Size_Expect_400M)
{
    UT_COMM_CREATE_DEFAULT(comm);
    void* buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(buffer != nullptr, true);
    EXPECT_EQ(size, 400 * MB_UNIT);

    void* secondGetBuffer = nullptr;
    uint64_t secondGetSize = 0;
    ret = HcclGetHcclBuffer(comm, &secondGetBuffer, &secondGetSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(secondGetBuffer == buffer, true);
    EXPECT_EQ(secondGetSize, 400 * MB_UNIT);
    Ut_Comm_Destroy(comm);
}

TEST_F(HcclIndependentOpMemTest, Ut_HcclGetHcclBuffer_When_Set_Mem_Size_Expect_By_Env)
{
    setenv("HCCL_BUFFSIZE", "1", 1);
    UT_COMM_CREATE_DEFAULT(comm);
    void* buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(buffer != nullptr, true);
    EXPECT_EQ(size, 2 * MB_UNIT);

    void* secondGetBuffer = nullptr;
    uint64_t secondGetSize = 0;
    ret = HcclGetHcclBuffer(comm, &secondGetBuffer, &secondGetSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(secondGetBuffer == buffer, true);
    EXPECT_EQ(secondGetSize, 2 * MB_UNIT);
    Ut_Comm_Destroy(comm);
}

HcclResult CreateCommByConfig(HcclComm* comm, uint64_t buffSize)
{
    Ut_Device_Set(0);
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = buffSize;
    HcclResult ret = HcclCommInitClusterInfoConfig(RANKTABLE_FILE_NAME, rankId, &commConfig, comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    return ret;
}

TEST_F(HcclIndependentOpMemTest, Ut_HcclGetHcclBuffer_When_Set_Mem_Size_Expect_Size_By_Config)
{
    setenv("HCCL_BUFFSIZE", "1", 1);
    uint64_t buffSize = 400;
    HcclResult ret = CreateCommByConfig(&comm, buffSize);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    void* buffer = nullptr;
    uint64_t size = 0;
    ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(buffer != nullptr, true);
    EXPECT_EQ(size, buffSize * 2 * MB_UNIT);

    void* secondGetBuffer = nullptr;
    uint64_t secondGetSize = 0;
    ret = HcclGetHcclBuffer(comm, &secondGetBuffer, &secondGetSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(secondGetBuffer == buffer, true);
    EXPECT_EQ(secondGetSize, buffSize * 2 * MB_UNIT);
    Ut_Comm_Destroy(comm);
}
