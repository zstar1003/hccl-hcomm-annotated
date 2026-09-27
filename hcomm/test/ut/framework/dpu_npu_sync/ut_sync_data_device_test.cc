/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../communicator/hccl_api_base_test.h"
#include "aicpu_ts_sync_data_c_adpt.h"

#include <chrono>
#include <thread>
#include "thread.h"
#include "aicpu_ts_thread.h"

using namespace hccl;

constexpr size_t SHMEM_SIZE_BYTE = 512;
constexpr size_t MSG_TAG_SIZE_BYTE = 256;
constexpr uint32_t TIMEOUT_SIZE_BYTE = 4;      // timeout字段长度(字节)
constexpr uint32_t CTRL_HDR_DATA_SIZE_LEN = 8; // data size字段长度(字节)

class SyncDataDeviceTest : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

#pragma pack(push)
#pragma pack(1)
    struct MsgHeader {
        uint8_t flag;
        char msgTag[MSG_TAG_SIZE_BYTE];
        uint32_t msgId;
    };
#pragma pack(pop)

protected:
    // 分配并清零一块模拟设备共享内存
    static void* AllocZeroedShmem()
    {
        void* devShmem = malloc(SHMEM_SIZE_BYTE);
        memset_s(devShmem, SHMEM_SIZE_BYTE, 0, SHMEM_SIZE_BYTE);
        return devShmem;
    }

    // 模拟 DPU kernel：向共享内存写入数据、msgTag、msgId 并置位 flag
    static void WriteDpuMsg(void* devShmem, const char* dpuData, size_t dpuDataSizeByte, uint32_t dpuMsgId)
    {
        MsgHeader* structedDevShmem = reinterpret_cast<MsgHeader*>(devShmem);
        strcpy_s(
            reinterpret_cast<char*>(structedDevShmem) + sizeof(MsgHeader) + TIMEOUT_SIZE_BYTE, dpuDataSizeByte,
            dpuData);
        strcpy_s(structedDevShmem->msgTag, MSG_TAG_SIZE_BYTE, "DPU Msg");
        structedDevShmem->msgId = dpuMsgId;
        structedDevShmem->flag = 1;
        printf("Dpu Kernel End.\n");
    }
};

TEST_F(SyncDataDeviceTest, ut_HcommSendRequest_When_Normal_Expect_ReturnIsHCCL_SUCCESS_And_MemoryIsCorrect)
{
    void* devShmem = malloc(SHMEM_SIZE_BYTE);

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    const char msgTag[MSG_TAG_SIZE_BYTE] = "Hello HCCL";
    const char data[] = "Open Source is Good.";
    const size_t dataSizeByte = sizeof(data);
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommSendRequest(handle, msgTag, data, dataSizeByte, &outMsgId);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    MsgHeader* structedDevShmem = static_cast<MsgHeader*>(devShmem);

    printf(
        "Simulated Device Shared Mem: [ %u | %s | %u | %s ]\n", structedDevShmem->flag, structedDevShmem->msgTag,
        structedDevShmem->msgId, static_cast<char*>(devShmem) + sizeof(MsgHeader));

    EXPECT_EQ(structedDevShmem->flag, 1);
    EXPECT_STREQ(structedDevShmem->msgTag, msgTag);
    EXPECT_EQ(structedDevShmem->msgId, outMsgId);
    EXPECT_STREQ(static_cast<char*>(devShmem) + sizeof(MsgHeader) + TIMEOUT_SIZE_BYTE + CTRL_HDR_DATA_SIZE_LEN, data);

    free(devShmem);
    devShmem = nullptr;
}

TEST_F(SyncDataDeviceTest, ut_HcommSendRequest_When_HandleIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    void* devShmem = nullptr;

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    const char msgTag[MSG_TAG_SIZE_BYTE] = "Hello HCCL";
    const char data[] = "Open Source is Good.";
    const size_t dataSizeByte = sizeof(data);
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommSendRequest(handle, msgTag, data, dataSizeByte, &outMsgId);

    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(SyncDataDeviceTest, ut_HcommWaitResponse_When_Normal_Expect_ReturnIsHCCL_SUCCESS_And_ResultIsCorrect)
{
    void* devShmem = AllocZeroedShmem();

    const char dpuData[] = "Open Source is Good.";
    const size_t dpuDataSizeByte = sizeof(dpuData);
    const uint32_t dpuMsgId = 1145;

    std::thread dpuKernel([=]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        WriteDpuMsg(devShmem, dpuData, dpuDataSizeByte, dpuMsgId);
    });

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    char dst[SHMEM_SIZE_BYTE] = "";
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommWaitResponse(handle, dst, dpuDataSizeByte, &outMsgId);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    printf("dst: %s\n", dst);

    EXPECT_STREQ(dst, dpuData);
    EXPECT_EQ(outMsgId, dpuMsgId);
    EXPECT_EQ(static_cast<MsgHeader*>(devShmem)->flag, 0); // flag is resetted to 0

    dpuKernel.join();

    free(devShmem);
    devShmem = nullptr;
}

TEST_F(SyncDataDeviceTest, ut_HcommThreadSynchronize_When_ThreadIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    ThreadHandle thread = 0;
    int32_t ret = HcommThreadSynchronize(thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(SyncDataDeviceTest, ut_HcommThreadSynchronize_When_ThreadIsValid_Expect_ReturnIsHCCL_SUCCESS)
{
    hccl::Thread* threadPtr = new (std::nothrow) hccl::AicpuTsThread(std::string());
    ASSERT_NE(threadPtr, nullptr);

    ThreadHandle thread = reinterpret_cast<ThreadHandle>(threadPtr);
    int32_t ret = HcommThreadSynchronize(thread);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete threadPtr;
    threadPtr = nullptr;
}

// HCCL_EXEC_TIMEOUT 配置为 0 时，GetSyncWaitTimeoutSeconds() 向上取整为 0，轮询不做超时判断，flag 稍后置位仍返回
// SUCCESS
TEST_F(SyncDataDeviceTest, ut_HcommWaitResponse_When_ExecTimeoutIsZero_Expect_WaitUntilFlagSetReturnSuccess)
{
    void* devShmem = AllocZeroedShmem();

    const char dpuData[] = "Open Source is Good.";
    const size_t dpuDataSizeByte = sizeof(dpuData);
    const uint32_t dpuMsgId = 7890;

    double execTimeout = 0.0;
    MOCKER(GetExternalInputHcclExecTimeOut).stubs().will(returnValue(execTimeout));

    std::thread dpuKernel([=]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        WriteDpuMsg(devShmem, dpuData, dpuDataSizeByte, dpuMsgId);
    });

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    char dst[SHMEM_SIZE_BYTE] = "";
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommWaitResponse(handle, dst, dpuDataSizeByte, &outMsgId);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(dst, dpuData);
    EXPECT_EQ(outMsgId, dpuMsgId);
    EXPECT_EQ(static_cast<MsgHeader*>(devShmem)->flag, 0);

    dpuKernel.join();
    GlobalMockObject::verify();

    free(devShmem);
    devShmem = nullptr;
}

// HCCL_EXEC_TIMEOUT 配置为 1 秒，flag 始终不被置位，轮询超过 1 秒后返回 HCCL_E_TIMEOUT
TEST_F(SyncDataDeviceTest, ut_HcommWaitResponse_When_FlagNotSetAndTimeout_Expect_ReturnHCCL_E_TIMEOUT)
{
    void* devShmem = AllocZeroedShmem();

    double execTimeout = 1.0;
    MOCKER(GetExternalInputHcclExecTimeOut).stubs().will(returnValue(execTimeout));

    MsgHandle handle = reinterpret_cast<MsgHandle>(devShmem);
    uint32_t outMsgId = 0;
    int32_t ret = HCCL_E_RESERVED;

    ret = HcommWaitResponse(handle, nullptr, 0, &outMsgId);

    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();

    free(devShmem);
    devShmem = nullptr;
}
