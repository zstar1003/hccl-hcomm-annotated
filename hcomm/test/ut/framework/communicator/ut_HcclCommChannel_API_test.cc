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

using namespace std;
using namespace hccl;

constexpr const char* RANKTABLE_FILE_NAME = "./ut_independent_op_test.json";

class HcclCommChannelTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_RANK_TABLE_910_1SERVER_1RANK;
        UT_COMM_CREATE_DEFAULT(comm);
    }
    void TearDown() override
    {
        Ut_Comm_Destroy(comm);
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclCommChannelTest, Ut_HcclChannelAcquire_When_Comm_Nullptr_Return_EPTR)
{
    HcclChannelDesc desc[2];
    HcclResult ret = HcclChannelDescInit(desc, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = CommProtocol::COMM_PROTOCOL_ROCE;
    desc[0].notifyNum = 2;
    desc[1].remoteRank = 2;
    desc[1].channelProtocol = CommProtocol::COMM_PROTOCOL_ROCE;
    desc[1].notifyNum = 2;

    ChannelHandle channel;
    ret = HcclChannelAcquire(nullptr, CommEngine::COMM_ENGINE_CPU, desc, 2, &channel);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
