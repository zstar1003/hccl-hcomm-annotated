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
#include "test_mock_setup.h"
#include "local_notify_v2.h"
#include "dev_capability.h"
#include "not_support_exception.h"
using namespace Hccl;

class RtsNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RtsNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RtsNotifyTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        SETUP_COMM_NOTIFY_MOCKS();
        std::cout << "A Test case in RtsNotifyTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RtsNotifyTest TearDown" << std::endl;
    }
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u32 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
    const u64 RDMA_SEND_MAX_SIZE = 0x80000000;  // 节点间RDMA发送数据单个WQE支持的最大数据量
    const u64 SDMA_SEND_MAX_SIZE = 0x100000000; // 节点内单个SDMA任务发送数据支持的最大数据量
    const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_910A = {
        {DataType::INT8, false},   {DataType::INT16, false},  {DataType::INT32, false},  {DataType::FP16, false},
        {DataType::FP32, true},    {DataType::INT64, false},  {DataType::UINT64, false}, {DataType::UINT8, false},
        {DataType::UINT16, false}, {DataType::UINT32, false}, {DataType::FP64, false},   {DataType::BFP16, false},
        {DataType::INT128, false},
    };
    const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_910A
        = {{ReduceOp::SUM, true}, {ReduceOp::PROD, false}, {ReduceOp::MAX, false}, {ReduceOp::MIN, false}};
    const u32 CAP_NOTIFY_SIZE_910A = 8;
    const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A = 128;
    const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_910A3 = {
        {DataType::INT8, true},    {DataType::INT16, true},   {DataType::INT32, true},   {DataType::FP16, true},
        {DataType::FP32, true},    {DataType::INT64, false},  {DataType::UINT64, false}, {DataType::UINT8, false},
        {DataType::UINT16, false}, {DataType::UINT32, false}, {DataType::FP64, false},   {DataType::BFP16, true},
        {DataType::INT128, false},
    };
    const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_910A3
        = {{ReduceOp::SUM, true}, {ReduceOp::PROD, false}, {ReduceOp::MAX, true}, {ReduceOp::MIN, true}};
    const u32 CAP_NOTIFY_SIZE_910A3 = 4;
    const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A3 = 32;

    const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_V82 = {
        {DataType::INT8, true},    {DataType::INT16, true},    {DataType::INT32, true},   {DataType::FP16, true},
        {DataType::FP32, true},    {DataType::INT64, false},   {DataType::UINT64, false}, {DataType::UINT8, true},
        {DataType::UINT16, true},  {DataType::UINT32, true},   {DataType::FP64, false},   {DataType::BFP16, true},
        {DataType::INT128, false}, {DataType::BF16_SAT, true},
    };
    const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_V82
        = {{ReduceOp::SUM, true},
           {ReduceOp::PROD, false},
           {ReduceOp::MAX, true},
           {ReduceOp::MIN, true},
           {ReduceOp::EQUAL, true}};
    const u32 CAP_NOTIFY_SIZE_V82 = 8;
    const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82 = 32;
};

const u32 CAP_NOTIFY_SIZE_V82 = 8;

TEST_F(RtsNotifyTest, rtsNotify_dev_used_false)
{
    Stream stream;
    RtsNotify notify(false);

    notify.Post(stream);
    notify.Wait(stream, 100);
    notify.Describe();

    EXPECT_EQ(fakeName, notify.SetIpcName());
    EXPECT_EQ(fakeNotifyId, notify.GetId());
    EXPECT_EQ(fakeOffset, notify.GetOffset());
    EXPECT_EQ(fakeNotifyHandleAddr, notify.GetHandleAddr());
    EXPECT_EQ(CAP_NOTIFY_SIZE_V82, notify.GetSize());
    EXPECT_EQ(false, notify.IsDevUsed());
    EXPECT_EQ(fakeDevPhyId, notify.GetDevPhyId());
}

TEST_F(RtsNotifyTest, rtsNotify_dev_used_true)
{
    Stream stream;
    RtsNotify notify(true);

    notify.Post(stream);
    notify.Wait(stream, 100);
    notify.Describe();

    EXPECT_EQ(fakeName, notify.SetIpcName());
    EXPECT_EQ(fakeNotifyId, notify.GetId());
    EXPECT_EQ(fakeOffset, notify.GetOffset());
    EXPECT_EQ(fakeNotifyHandleAddr, notify.GetHandleAddr());
    EXPECT_EQ(CAP_NOTIFY_SIZE_V82, notify.GetSize());
    EXPECT_EQ(true, notify.IsDevUsed());
    EXPECT_EQ(fakeDevPhyId, notify.GetDevPhyId());
}
