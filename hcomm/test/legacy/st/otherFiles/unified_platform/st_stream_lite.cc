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

#define private public

#include "stream_lite.h"
#include "ins_to_sqe_rule.h"
#include "binary_stream.h"
#include "string_util.h"
#include "rtsq_a5.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"
#include "local_ub_rma_buffer.h"
#include "ub_transport_lite_impl.h"
#include "ub_mem_transport.h"
#include "mem_transport_lite.h"

#undef private

using namespace Hccl;
class StreamLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StreamLite tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "StreamLite tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

        std::cout << "A Test case in StreamLite SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in StreamLite TearDown" << std::endl;
    }
    u32 fakeStreamId = 1;
    u32 fakeSqId = 2;
    u32 fakedevPhyId = 3;

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(StreamLiteTest, stream_lite_given_uniqueId)
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
