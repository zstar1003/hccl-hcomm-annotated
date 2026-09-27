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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "aicpu_ts_thread.h"
#include "aicpu_ts_thread_interface.h"
#include "dispatcher_aicpu.h"

class UtAicpuTsBase : public testing::Test {
protected:
    virtual void SetUp() override
    {
        bool isDeviceSide = false;
        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

        threadOnDevice = std::make_unique<hccl::AicpuTsThread>(
            hccl::StreamType::STREAM_TYPE_DEVICE, 2, hccl::NotifyLoadType::DEVICE_NOTIFY);

        HcclResult ret = threadOnDevice->Init();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        std::string mainStr = threadOnDevice->GetUniqueId();

        GlobalMockObject::reset();

        isDeviceSide = true;
        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

        mainDevThread = std::make_unique<hccl::AicpuTsThread>(mainStr);
        ret = mainDevThread->Init();
        std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
            = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
                  return HCCL_SUCCESS;
              };
        mainDevThread->SetAddTaskInfoCallback(callback);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        thread = reinterpret_cast<ThreadHandle>(mainDevThread.get());

        MOCKER_CPP(
            &Hccl::IAicpuTsThread::SdmaCopy, HcclResult(Hccl::IAicpuTsThread::*)(uint64_t, uint64_t, uint64_t) const)
            .stubs()
            .will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    std::unique_ptr<hccl::AicpuTsThread> threadOnDevice;
    std::unique_ptr<hccl::AicpuTsThread> mainDevThread;
    ThreadHandle thread;
};
