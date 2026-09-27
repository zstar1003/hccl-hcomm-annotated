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
#define private public
#define protected public
#include "dev_type.h"
#include "dev_capability.h"
#include "orion_adapter_rts.h"
#include "not_support_exception.h"
#include "env_config_stub.h"
#include "env_config_v2.h"
#undef protected
#undef private

using namespace Hccl;

std::map<std::string, std::string> envCfgMapStub1 = defaultEnvCfgMap;

char* getenv_stub_func1(const char* __name)
{
    char* ret = const_cast<char*>(envCfgMapStub1[std::string(__name)].c_str());
    return ret;
}

class DevCapabilityTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevCapability tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DevCapability tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in DevCapability SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DevCapability TearDown" << std::endl;
    }
    void MockFunc()
    {
        MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(getenv_stub_func1));

        char c = '1';
        MOCKER(realpath).stubs().with(mockcpp::any()).will(returnValue(&c));

        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A));
    }

    void ResetEnvCfgMap()
    {
        envCfgMapStub1.clear();
        envCfgMapStub1 = defaultEnvCfgMap;
    }
    const u32 CAP_NOTIFY_SIZE_V82 = 8;
    const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82 = 32;
    const u64 RDMA_SEND_MAX_SIZE = 0x80000000;  // 节点间RDMA发送数据单个WQE支持的最大数据量
    const u64 SDMA_SEND_MAX_SIZE = 0x100000000; // 节点内单个SDMA任务发送数据支持的最大数据量
    const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_V82
        = {{ReduceOp::SUM, true},
           {ReduceOp::PROD, false},
           {ReduceOp::MAX, true},
           {ReduceOp::MIN, true},
           {ReduceOp::EQUAL, true}};
    const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_V82 = {
        {DataType::INT8, true},    {DataType::INT16, true},    {DataType::INT32, true},   {DataType::FP16, true},
        {DataType::FP32, true},    {DataType::INT64, false},   {DataType::UINT64, false}, {DataType::UINT8, true},
        {DataType::UINT16, true},  {DataType::UINT32, true},   {DataType::FP64, false},   {DataType::BFP16, true},
        {DataType::INT128, false}, {DataType::BF16_SAT, true},
    };
};

const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_V82
    = {{ReduceOp::SUM, true},
       {ReduceOp::PROD, false},
       {ReduceOp::MAX, true},
       {ReduceOp::MIN, true},
       {ReduceOp::EQUAL, true}};

const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_V82 = {
    {DataType::INT8, true},    {DataType::INT16, true},    {DataType::INT32, true},   {DataType::FP16, true},
    {DataType::FP32, true},    {DataType::INT64, false},   {DataType::UINT64, false}, {DataType::UINT8, true},
    {DataType::UINT16, true},  {DataType::UINT32, true},   {DataType::FP64, false},   {DataType::BFP16, true},
    {DataType::INT128, false}, {DataType::BF16_SAT, true},
};

const u32 CAP_NOTIFY_SIZE_V82 = 8;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82 = 32;

const u64 RDMA_SEND_MAX_SIZE = 0x80000000;  // 节点间RDMA发送数据单个WQE支持的最大数据量
const u64 SDMA_SEND_MAX_SIZE = 0x100000000; // 节点内单个SDMA任务发送数据支持的最大数据量

TEST_F(DevCapabilityTest, test_dev_cap_v82)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    ResetEnvCfgMap();
    MockFunc();

    DevCapability& devCap = DevCapability::GetInstance();
    EXPECT_EQ(CAP_NOTIFY_SIZE_V82, devCap.GetNotifySize());
    EXPECT_EQ(SDMA_SEND_MAX_SIZE, devCap.GetSdmaSendMaxSize());
    EXPECT_EQ(RDMA_SEND_MAX_SIZE, devCap.GetRdmaSendMaxSize());
    EXPECT_EQ(CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82, devCap.GetSdmaInlineReduceAlignBytes());
    EXPECT_EQ(CAP_INLINE_REDUCE_OP_V82, devCap.GetInlineReduceOpMap());
    EXPECT_EQ(CAP_INLINE_REDUCE_DATATYPE_V82, devCap.GetInlineReduceDataTypeMap());
    EXPECT_EQ(true, devCap.IsSupportWriteWithNotify());
    EXPECT_EQ(true, devCap.IsSupportStarsPollNetCq());
    EXPECT_EQ(true, devCap.IsSupportDevNetInlineReduce());
}

TEST_F(DevCapabilityTest, Ut_Load910A3Cap_When_910A3_Expect_ReturnIsTrue)
{
    DevType devType = DevType::DEV_TYPE_910A3;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    ResetEnvCfgMap();
    MockFunc();

    DevCapability& devCap = DevCapability::GetInstance();
    devCap.Load910A3Cap();

    EXPECT_EQ(true, devCap.IsSupportDevNetInlineReduce());
}

TEST_F(DevCapabilityTest, Ut_Load910ACap_When_910A3_Expect_ReturnIsFalse)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    ResetEnvCfgMap();
    MockFunc();

    DevCapability& devCap = DevCapability::GetInstance();
    devCap.Load910ACap();
    EXPECT_EQ(false, devCap.IsSupportDevNetInlineReduce());
}

TEST_F(DevCapabilityTest, Ut_DevCapabilityT_Init)
{
    DevType devType = DevType::DEV_TYPE_V51_310_P3;
    DevCapability& devCap = DevCapability::GetInstance();
    devCap.Reset();
    EXPECT_THROW(devCap.Init(devType), NotSupportException);
}
