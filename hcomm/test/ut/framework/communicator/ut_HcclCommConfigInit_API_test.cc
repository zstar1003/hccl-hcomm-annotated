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

class HcclCommConfigInitTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_1RANK_AS_DEFAULT;
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclCommConfigInitTest, Ut_HcclCommConfigInit_When_Normal_Expect_Normal)
{
    Ut_Device_Set(0);
    const char* rankTableFile = rankTableFileName;
    u32 rankId = 0;
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    const auto* configInfo = reinterpret_cast<const CommConfigInfo*>(commConfig.reserved);
    EXPECT_EQ(configInfo->version, 11U);
    EXPECT_EQ(configInfo->configSize, sizeof(HcclCommConfig));
    EXPECT_EQ(commConfig.hcclChannelSqDepth, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);
    commConfig.hcclBufferSize = 400;
    HcclResult ret = HcclCommInitClusterInfoConfig(rankTableFile, rankId, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}
