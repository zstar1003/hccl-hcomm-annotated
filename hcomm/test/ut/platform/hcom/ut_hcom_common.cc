/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include <iostream>
#include <algorithm>
#include <list>
#include <vector>
#include <string>
#include <securec.h>
#include <hccl/hccl_types.h>
#include "config.h"
#include "hccl/base.h"
#include "param_check_pub.h"
#include "../op_base/src/op_base.h"
#include "hccl/hcom.h"
#include "hcom_common.h"
#include "rank_consistentcy_checker.h"
#include "profiling_manager_pub.h"
#include "topoinfo_ranktableParser_pub.h"
#include "stream_pub.h"

#include "comm_base_pub.h"
#include "topoinfo_ranktableOffline.h"
#include "sal.h"
#include "mmpa_api.h"
#include "coll_alg_utils.h"
#include "json_utils.h"

#include "hcom_pub.h"
#include "hccl_comm_pub.h"
#include "hccl_communicator.h"

using namespace std;
using namespace hccl;

class HcomCommonTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcomCommon_UT SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcomCommon_UT TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(HcomCommonTest, ut_hcom_common_HcomGetSecAddrCopyFlag)
{
    bool ret;

    ret = HcomGetSecAddrCopyFlag("Ascend910B");
    EXPECT_EQ(ret, true);
}

TEST_F(HcomCommonTest, ut_hcom_HcomCreateCommCCLbuffer)
{
    MOCKER(HcomCheckGroupName).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetInCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetOutCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));

    std::shared_ptr<hccl::hcclComm> h = std::make_shared<hccl::hcclComm>();
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(h)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::CreateCommCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));

    const char* group = "123";
    HcclResult ret = HcomCreateCommCCLbuffer(group);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* buffer = nullptr;
    u64 size = 0;
    ret = HcomGetInCCLbuffer(group, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomGetOutCCLbuffer(group, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomCommonTest, ut_hcom_common_HcomGetAicpuOpStreamNotify)
{
    std::string identifier = "aa";
    HcclRtStream opStream;
    void* aicpuNotify;
    HcclResult ret;

    MOCKER(HcomCheckGroupName).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(HcclGetCommHandle).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetAicpuOpStreamNotify).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    ret = HcomGetAicpuOpStreamNotify(identifier.c_str(), &opStream, 1, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomCommonTest, ut_hcom_common_HcomMc2AiCpuStreamAllocAndGet)
{
    std::string identifier = "aa";
    rtStream_t aiCpuStream;
    HcclResult ret;

    MOCKER(HcomCheckGroupName).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER(HcclGetCommHandle).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::Mc2AiCpuStreamAllocAndGet).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    ret = HcomMc2AiCpuStreamAllocAndGet(identifier.c_str(), 1, &aiCpuStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}
