/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_aicpu_ts_base.h"
#include "ub_transport_lite_impl.h"
#include "transport_pub.h"

using namespace hccl;

class UtAicpuTsHcommBatchTransferOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommBatchTransferOnThread tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsHcommBatchTransferOnThread tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommBatchTransferOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommBatchTransferOnThread TearDown" << std::endl;
    }

    uint64_t tempDst[6] = {0};
    uint64_t tempSrc[6] = {1, 1, 4, 5, 1, 4};
    void* dst = reinterpret_cast<void*>(tempDst);
    void* src = reinterpret_cast<void*>(tempSrc);
    uint64_t len = sizeof(tempDst);
    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportDev{uniqueId};
    ChannelHandle devHandle = reinterpret_cast<ChannelHandle>(&transportDev);
    int32_t res{HCCL_E_RESERVED};
    HcommBatchTransferDesc transferDescs[2];
    uint32_t transferDescNum = 4;
    std::vector<HcommBatchTransferDesc> transferDescVec;

private:
    void GenerateTransferDescs(
        HcommBatchTransferDesc tmpSingleDesc, uint32_t transferDescNum,
        std::vector<HcommBatchTransferDesc>& transferDescVec)
    {
        for (uint8_t i = 0; i < transferDescNum; ++i) {
            transferDescVec.push_back(tmpSingleDesc);
        }
    }
};

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_ThreadIsNull_Expect_ReturnHCCL_E_PTR)
{
    transferDescs[0].transferInfo.write.len = len;
    transferDescs[0].transferInfo.write.dst = dst;
    transferDescs[0].transferInfo.write.src = src;
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;

    res = HcommBatchTransferOnThread(0, devHandle, transferDescs, 1);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_DescsIsNull_Expect_ReturnHCCL_E_PTR)
{
    res = HcommBatchTransferOnThread(thread, devHandle, nullptr, 1);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_DescNumIsZero_Expect_ReturnHCCL_E_PARA)
{
    transferDescs[0].transferInfo.write.len = len;
    transferDescs[0].transferInfo.write.dst = dst;
    transferDescs[0].transferInfo.write.src = src;
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescs, 0);
    EXPECT_EQ(res, HCCL_E_PARA);
}

TEST_F(
    UtAicpuTsHcommBatchTransferOnThread,
    Ut_HcommBatchTransferOnThread_When_InvalidTransType_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    transferDescs[0].transferInfo.write.len = len;
    transferDescs[0].transferInfo.write.dst = dst;
    transferDescs[0].transferInfo.write.src = src;
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_INVALID;

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescs, 1);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_DstIsNull_Expect_ReturnHCCL_E_PTR)
{
    transferDescs[0].transferInfo.write.len = len;
    transferDescs[0].transferInfo.write.dst = nullptr;
    transferDescs[0].transferInfo.write.src = src;
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescs, 1);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_SrcIsNull_Expect_ReturnHCCL_E_PTR)
{
    transferDescs[0].transferInfo.write.len = len;
    transferDescs[0].transferInfo.write.dst = dst;
    transferDescs[0].transferInfo.write.src = nullptr;
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescs, 1);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtAicpuTsHcommBatchTransferOnThread,
    Ut_HcommBatchTransferOnThread_When_MultipleDescs_Expect_ReturnHCCL_HCCL_SUCCESS)
{
    transferDescs[0].transferInfo.write.len = len;
    transferDescs[0].transferInfo.write.dst = dst;
    transferDescs[0].transferInfo.write.src = src;
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;

    transferDescs[1].transferInfo.read.len = len;
    transferDescs[1].transferInfo.read.dst = dst;
    transferDescs[1].transferInfo.read.src = src;
    transferDescs[1].transType = HCOMM_TRANSFER_TYPE_READ;

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescs, 2);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_channel_IsNull_Expect_PtrError)
{
    ChannelHandle channel = 0;
    HcommBatchTransferDesc* transferDescs = reinterpret_cast<HcommBatchTransferDesc*>(0x1000);
    uint32_t transferDescNum = 1;

    res = HcommBatchTransferOnThread(thread, channel, transferDescs, transferDescNum);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_Write_Expect_Success)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE;
    tmpTransferDesc.transferInfo.write.dst = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.write.src = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.write.len = 64;
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_Read_Expect_Success)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_READ;
    tmpTransferDesc.transferInfo.write.dst = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.write.src = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.write.len = 64;
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_WriteReduce_Expect_Success)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE_REDUCE;
    tmpTransferDesc.transferInfo.reduce.dst = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.reduce.src = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.reduce.reduceOp = HCOMM_REDUCE_SUM;
    tmpTransferDesc.transferInfo.reduce.dataType = HCOMM_DATA_TYPE_INT64;
    tmpTransferDesc.transferInfo.reduce.count = 64;
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_ReadReduce_Expect_Success)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_READ_REDUCE;
    tmpTransferDesc.transferInfo.reduce.dst = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.reduce.src = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.reduce.reduceOp = HCOMM_REDUCE_SUM;
    tmpTransferDesc.transferInfo.reduce.dataType = HCOMM_DATA_TYPE_INT64;
    tmpTransferDesc.transferInfo.reduce.count = 64;
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_NotifyRecord_Expect_Success)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    tmpTransferDesc.transferInfo.notifyRecord.notifyIdx = 10;
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_WriteWithNotify_Expect_Success)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY;
    tmpTransferDesc.transferInfo.writeWithNotify.notifyIdx = 10;
    tmpTransferDesc.transferInfo.writeWithNotify.dst = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.writeWithNotify.src = reinterpret_cast<void*>(0x1000);
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_WriteReduceWithNotify_Expect_Success)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.notifyIdx = 10;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.dst = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.writeReduceWithNotify.src = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.writeReduceWithNotify.dataType = HCOMM_DATA_TYPE_INT64;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.reduceOp = HCOMM_REDUCE_SUM;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.count = 64;
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommBatchTransferOnThread, Ut_HcommBatchTransferOnThread_When_NotifyIdxIsInvaild_Expect_E_PARA)
{
    HcommBatchTransferDesc tmpTransferDesc;
    tmpTransferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.notifyIdx = 0xFFFFFFFF;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.dst = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.writeReduceWithNotify.src = reinterpret_cast<void*>(0x1000);
    tmpTransferDesc.transferInfo.writeReduceWithNotify.dataType = HCOMM_DATA_TYPE_INT64;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.reduceOp = HCOMM_REDUCE_SUM;
    tmpTransferDesc.transferInfo.writeReduceWithNotify.count = 64;
    GenerateTransferDescs(tmpTransferDesc, transferDescNum, transferDescVec);

    res = HcommBatchTransferOnThread(thread, devHandle, transferDescVec.data(), transferDescNum);
    EXPECT_EQ(res, HCCL_E_PARA);
}
