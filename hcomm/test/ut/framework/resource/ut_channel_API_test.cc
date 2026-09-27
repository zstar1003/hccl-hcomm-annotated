/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "hccl_api_base_test.h"
#include "channel_factory.h"
#include "dpu_roce_channel.h"
#include "hccl/hccl_res.h"
#include "channel_manager.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include "mockcpp/mockcpp.hpp"
// #include "gmock/gmock.h"

#define private public

using namespace hccl;

class CommChannelGetStatusTest : public BaseInit {
public:
    HcclComm comm;
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
