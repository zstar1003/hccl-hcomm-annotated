/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator.h"
#include "hccl_api_base_test.h"
#include "hcom_pub.h"
#include "hccl_comm_pub.h"

// 测试类定义
class HcclGetStreamNumforAivModeTest : public BaseInit {
protected:
    // Common variables
    u64 streamNum = 0;
    u64 dataSize = 1024;
    u64 count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    u32 aivCoreLimit = 48;
    HcclCMDType optype = HCCL_CMD_ALLREDUCE;
    std::string algName = "";
    bool ifAiv = false;
    HcclAlg* implAlg_ = nullptr;
    HcclCommunicator communicator_;

    void SetUp() override
    {
        BaseInit::SetUp();
        static CCLBufferManager mockBufferManager;
        static HcclDispatcher mockDispatcher = nullptr;

        // 初始化 implAlg_
        implAlg_ = new HcclAlg(mockBufferManager, mockDispatcher, mockDispatcher);
        MOCKER_CPP_VIRTUAL(communicator_, &HcclCommunicator::GetAlgType)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
    }

    void TearDown() override
    {
        delete implAlg_;
        implAlg_ = nullptr;
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

// 正常路径测试 1: 正常调用，使用默认组
TEST_F(HcclGetStreamNumforAivModeTest, UT_HcclGetStreamNumforAivMode_When_DefaultGroup_Expect_Success)
{
    u64 dataSize = 1024;
    u64 count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    u32 aivCoreLimit = 4;
    HcclCMDType optype = HCCL_CMD_ALLREDUCE;

    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::GetWorkspaceSubStreamNum).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret
        = HcomGetWorkspaceSubStreamNum(nullptr, streamNum, dataSize, dataType, aivCoreLimit, reduceOp, count, optype);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 正常路径测试 2: 正常调用，使用自定义组
TEST_F(HcclGetStreamNumforAivModeTest, UT_HcclGetStreamNumforAivMode_When_CustomGroup_Expect_Success)
{
    u64 dataSize = 2048;
    u64 count = 20;
    HcclDataType dataType = HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HCCL_REDUCE_MAX;
    u32 aivCoreLimit = 2;
    HcclCMDType optype = HCCL_CMD_REDUCE_SCATTER;

    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetWorkspaceSubStreamNum).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcomGetWorkspaceSubStreamNum(
        "custom_group", streamNum, dataSize, dataType, aivCoreLimit, reduceOp, count, optype);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 正常路径测试 3: 正常调用，边界数据
TEST_F(HcclGetStreamNumforAivModeTest, UT_HcclGetStreamNumforAivMode_When_BoundaryData_Expect_Success)
{
    u64 dataSize = 0;
    u64 count = 1;
    HcclDataType dataType = HCCL_DATA_TYPE_FP16;
    HcclReduceOp reduceOp = HCCL_REDUCE_MIN;
    u32 aivCoreLimit = 1;
    HcclCMDType optype = HCCL_CMD_ALLGATHER;

    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetWorkspaceSubStreamNum).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcomGetWorkspaceSubStreamNum(
        "boundary_group", streamNum, dataSize, dataType, aivCoreLimit, reduceOp, count, optype);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 边界条件测试 1: group 为 nullptr
TEST_F(HcclGetStreamNumforAivModeTest, UT_HcclGetStreamNumforAivMode_When_NullGroup_Expect_Success)
{
    u64 dataSize = 1024;
    u64 count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    u32 aivCoreLimit = 4;
    HcclCMDType optype = HCCL_CMD_ALLREDUCE;

    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetWorkspaceSubStreamNum).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret
        = HcomGetWorkspaceSubStreamNum(nullptr, streamNum, dataSize, dataType, aivCoreLimit, reduceOp, count, optype);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 可靠性用例: 极端条件下调用函数
TEST_F(
    HcclGetStreamNumforAivModeTest, UT_GetWorkspaceSubStreamNum_When_ReliabilityExtremeConditions_Expect_HCCL_SUCCESS)
{
    u64 count = 1e3;
    u64 dataSize = 1e6;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    HcclCMDType opType = HCCL_CMD_ALLREDUCE;
    std::string algName = "";
    bool ifAiv = true;

    MOCKER(GetAlgType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclGetCommHandle).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = communicator_.GetWorkspaceSubStreamNum(
        count, dataType, reduceOp, algName, streamNum, dataSize, ifAiv, opType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, 0);
}

// 安全性用例: 输入参数无效
TEST_F(HcclGetStreamNumforAivModeTest, UT_GetWorkspaceSubStreamNum_When_SafetyInvalidInputs_Expect_HCCL_SUCCESS)
{
    u64 count = 0;
    u64 dataSize = 0;
    HcclDataType dataType = static_cast<HcclDataType>(-1);
    HcclReduceOp reduceOp = static_cast<HcclReduceOp>(-1);
    HcclCMDType opType = static_cast<HcclCMDType>(-1);
    std::string algName = "";
    bool ifAiv = false;

    HcclResult ret = communicator_.GetWorkspaceSubStreamNum(
        count, dataType, reduceOp, algName, streamNum, dataSize, ifAiv, opType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 兼容性用例: 使用原始参数调用函数，确保向后兼容但是不向前兼容
TEST_F(
    HcclGetStreamNumforAivModeTest,
    UT_GetWorkspaceSubStreamNum_When_CompatibilityOriginalParameters_Expect_HCCL_SUCCESS)
{
    u64 dataSize = 1024;
    HcclCMDType opType = HCCL_CMD_ALLREDUCE;
    HcclResult ret = communicator_.GetWorkspaceSubStreamNum(
        0, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, "", streamNum, dataSize, false, opType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// AICPU 模式：mock HcclSelectAlg 设置 ifAiv=false，验证 streamNum 申请成功
// AICPU(ifAiv=false)走传统 deviceType 多流分支，streamNum 非零
TEST_F(HcclGetStreamNumforAivModeTest, UT_HcomGetWorkspaceSubStreamNum_When_AicpuMode_Expect_Success)
{
    // mock HcclGetCommHandle：显式构造非空 comm 出参，使 hcclComm 非空、过 null 检查直达 HcclSelectAlg
    MOCKER(HcclGetCommHandle)
        .stubs()
        .will(invoke(+[](const char*, std::shared_ptr<hccl::hcclComm>& comm) -> HcclResult {
            comm = std::make_shared<hccl::hcclComm>();
            return HCCL_SUCCESS;
        }));

    // mock HcclSelectAlg：第7个出参 ifAiv=false，走 AICPU 分支
    MOCKER_CPP(&hcclComm::HcclSelectAlg)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            outBound(false))
        .will(returnValue(HCCL_SUCCESS));

    // mock GetWorkspaceSubStreamNum：AICPU 走传统多流分支，第5个出参 streamNum 回写非零值 8
    MOCKER_CPP(&hcclComm::GetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(8ULL))
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret
        = HcomGetWorkspaceSubStreamNum(nullptr, streamNum, dataSize, dataType, aivCoreLimit, reduceOp, count, optype);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, 8ULL) << "AICPU 模式走传统多流分支，streamNum 应为非零值 8";
}

// AIV 模式：mock HcclSelectAlg 设置 ifAiv=true，验证 streamNum 申请成功
// AIV(Hcom图模式)展开走单流、不需额外从流，streamNum 业务事实上为 0
TEST_F(HcclGetStreamNumforAivModeTest, UT_HcomGetWorkspaceSubStreamNum_When_AivMode_Expect_Success)
{
    // mock HcclGetCommHandle：显式构造非空 comm 出参，使 hcclComm 非空、过 null 检查直达 HcclSelectAlg
    MOCKER(HcclGetCommHandle)
        .stubs()
        .will(invoke(+[](const char*, std::shared_ptr<hccl::hcclComm>& comm) -> HcclResult {
            comm = std::make_shared<hccl::hcclComm>();
            return HCCL_SUCCESS;
        }));

    // mock HcclSelectAlg：第7个出参 ifAiv=true，走 AIV 分支
    MOCKER_CPP(&hcclComm::HcclSelectAlg)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            outBound(true))
        .will(returnValue(HCCL_SUCCESS));

    // mock GetWorkspaceSubStreamNum：Hcom图模式+AIV 展开走单流，第5个出参 streamNum 回写 0
    MOCKER_CPP(&hcclComm::GetWorkspaceSubStreamNum)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(0ULL))
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret
        = HcomGetWorkspaceSubStreamNum(nullptr, streamNum, dataSize, dataType, aivCoreLimit, reduceOp, count, optype);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(streamNum, 0ULL) << "Hcom图模式+AIV 展开走单流，streamNum 应为 0";
}
