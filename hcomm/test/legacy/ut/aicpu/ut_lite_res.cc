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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>

#define private public
#include "stream_lite.h"
#include "ins_to_sqe_rule.h"
#include "binary_stream.h"
#include "rtsq_a5.h"
#include "dev_ub_connection.h"
#include "notify_lite.h"
#include "udma_data_struct.h"
#include "ub_conn_lite.h"
#include "mem_transport_lite.h"
#include "orion_adapter_rts.h"

#undef private

using namespace Hccl;

class LiteResTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LiteResTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "LiteResTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        std::cout << "A Test case in LiteResTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LiteResTest TearDown" << std::endl;
    }
    u32 fakeStreamId = 0;
    u32 fakeSqId = 0;
    u32 fakedevPhyId = 0;

    u32 fakeNotifyId = 1;
    u32 fakeNotifyDevPhyId = 1;

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(LiteResTest, test_stream_lite)
{
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakedevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    StreamLite stream(uniqueId);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    EXPECT_EQ(fakeStreamId, stream.GetId());
    EXPECT_EQ(fakeSqId, stream.GetSqId());
    EXPECT_EQ(fakedevPhyId, stream.GetDevPhyId());
    stream.Describe();
}

TEST_F(LiteResTest, test_notify_lite)
{
    BinaryStream binaryStream;
    binaryStream << fakeNotifyId;
    binaryStream << fakeNotifyDevPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);

    NotifyLite lite(result);
    EXPECT_EQ(fakeNotifyId, lite.GetId());
    EXPECT_EQ(fakeNotifyDevPhyId, lite.GetDevPhyId());
    lite.Describe();
}
