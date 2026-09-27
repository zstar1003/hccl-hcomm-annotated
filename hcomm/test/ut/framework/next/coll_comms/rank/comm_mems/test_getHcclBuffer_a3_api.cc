/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../hccl_api_base_test.h"
#include "hcomm_c_adpt.h"
#include "local_notify_impl.h"
#include "aicpu_launch_manager.h"
#include "llt_hccl_stub_rank_graph.h"

class TestHcclGetHcclBufferA3 : public BaseInit {
public:
    void SetUp() override
    {
        GlobalMockObject::verify();
        BaseInit::SetUp();
        const char* fakeA3SocName = "Ascend910_9362";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA3SocName));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(TestHcclGetHcclBufferA3, Ut_HcclGetHcclBufferA3_When_Normal_Return_HCCL_Success)
{
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    HcclComm commHandle;
    UT_USE_RANK_TABLE_910_1SERVER_1RANK;
    UT_COMM_CREATE_DEFAULT(commHandle);

    void* buffer;
    uint64_t size;
    HcclResult ret = HcclGetHcclBuffer(commHandle, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(commHandle);
    GlobalMockObject::verify();
}
