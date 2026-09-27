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
#include "mask_event.h"
#include "runtime_api_exception.h"
#include "invalid_params_exception.h"
using namespace Hccl;

class MaskEventTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdapterRts tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AdapterRts tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AdapterRts SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AdapterRts TearDown" << std::endl;
    }
};

TEST_F(MaskEventTest, ConstructMaskEvent_ok)
{
    // Given
    RtEvent_t event = new int(0);
    MOCKER(HrtEventCreateWithFlag).stubs().will(returnValue(event));

    // when
    MaskEvent* maskEvent = new MaskEvent();
    delete maskEvent;
    delete event;
}

TEST_F(MaskEventTest, MaskEventRecord_ok)
{
    // Given
    RtEvent_t event = nullptr;
    MOCKER(HrtEventCreateWithFlag).stubs().will(returnValue(event));

    MOCKER(HrtGetDevice).stubs().will(returnValue(1));

    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));

    // when
    MaskEvent maskEvent;
    Stream stream;

    // then
    EXPECT_NO_THROW(maskEvent.Record(stream));
}

TEST_F(MaskEventTest, MaskEventQueryStatus_ok)
{
    // Given
    HrtEventStatus status1 = HrtEventStatus::EVENT_INIT;
    HrtEventStatus status2 = HrtEventStatus::EVENT_RECORDED;
    MOCKER(HrtEventQueryStatus).stubs().will(returnValue(status1)).then(returnValue(status2));

    // when
    MaskEvent maskEvent;

    // then
    EXPECT_EQ(maskEvent.QueryStatus(), HrtEventStatus::EVENT_INIT);
    EXPECT_EQ(maskEvent.QueryStatus(), HrtEventStatus::EVENT_RECORDED);
}
