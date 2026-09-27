/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif
#include "stream_utils.h"
#include "env_config.h"
#include "op_base.h"
#include "aicpu_launch_manager.h"
#include "aicpu_thread_kernel.h"
#include "aicpu_communicator.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu.h"
#include "coll_comm_aicpu_kernel.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu_kernel_adpt.h"
#include "threads/thread_aicpu_mgr.h"
#include "aicpu_thread_process.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class Test_Engine_Aicpu_Interface : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Test_Engine_Aicpu_Interface SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "Test_Engine_Aicpu_Interface TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(Test_Engine_Aicpu_Interface, test_RunAicpuThreadSupplementNotify)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();

    ret = aicpuThread.SupplementNotify(4);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string supplementStr = aicpuThread.GetUniqueId();
    GlobalMockObject::verify();

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    isDeviceSide = true;
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CommAicpuParam aicpuPara{};
    std::string hcomId = "abcd";
    EXPECT_EQ(memcpy_s(aicpuPara.hcomId, sizeof(aicpuPara.hcomId), hcomId.c_str(), hcomId.size()), 0);
    aicpuPara.deviceLogicId = 0;
    aicpuPara.devicePhyId = 0;
    aicpuPara.deviceType = (u32)DevType::DEV_TYPE_950;
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    // RegisterThreadAddDfxTaskInfo 已迁入 ThreadAicpuMgr
    MOCKER_CPP(&ThreadAicpuMgr::RegisterThreadAddDfxTaskInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ThreadAicpuMgr::RegisterThreadCacheCallback).stubs().will(returnValue(HCCL_SUCCESS));
    void* args = (void*)(&aicpuPara);
    EXPECT_EQ(RunAicpuCommInit(args), 0);

    // InitAicpuIndOp 被 mock，手动初始化 commEngineResMgr_
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    for (auto& kv : commInfo) {
        if (kv.second->commEngineResMgr_ == nullptr) {
            kv.second->commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(kv.second->dfx_, [](bool) {
                return HCCL_SUCCESS;
            });
        }
    }

    ThreadMgrAicpuParam para{};
    para.threadNum = 1;
    para.deviceHandle = malloc(sizeof(ThreadHandle));
    size_t copyLen = std::min(mainStr.size(), static_cast<size_t>(THREAD_UNIQUE_ID_MAX_SIZE));
    EXPECT_EQ(memcpy_s(para.hcomId, sizeof(para.hcomId), hcomId.c_str(), hcomId.size()), 0);
    EXPECT_EQ(memcpy_s(para.threadParam[0], THREAD_UNIQUE_ID_MAX_SIZE, mainStr.c_str(), copyLen), 0);

    u64 addr = (u64)(&para);
    args = (void*)(&addr);
    EXPECT_EQ(RunAicpuIndOpThreadInit(args), 0);
    copyLen = std::min(supplementStr.size(), static_cast<size_t>(THREAD_UNIQUE_ID_MAX_SIZE));
    EXPECT_EQ(memcpy_s(para.threadParam[0], THREAD_UNIQUE_ID_MAX_SIZE, supplementStr.c_str(), copyLen), 0);
    EXPECT_EQ(RunAicpuThreadSupplementNotify(args), 0);

    MOCKER(CollCommAicpuKernelAdptInitNotify).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadProcess::AicpuThreadInit).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadProcess::AicpuThreadDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(RunAicpuIndOpNotify(args), 0);
    EXPECT_EQ(RunAicpuThreadInit(args), 0);
    EXPECT_EQ(RunAicpuThreadDestroy(args), 0);
    free(para.deviceHandle);
}
