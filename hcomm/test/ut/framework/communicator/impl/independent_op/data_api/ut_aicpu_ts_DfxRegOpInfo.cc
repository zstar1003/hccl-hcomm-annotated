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

#include "dfx_profiling_handler_lite.h"
#include "hccl_diag.h"
#include "llt_hccl_stub_pub.h"
#include "sqe_build_a5.h"
#include <thread>

using namespace hccl;

// 说明：设备类型无法用mockcpp构造——报到口对hrtGetDeviceType的调用被链接器直接绑定（不走PLT），
// hook机制无效。改用set_chip_type_stub设置芯片类型走真实链路：aclrtGetSocName返回"Ascend960"，
// 由__hrtGetDeviceType的socName匹配得出DEV_TYPE_960。__hrtGetDeviceType内有thread_local缓存g_deviceType，
// 故在全新线程中调用报到口保证按构造值重新查询；g_sqeProfBit同为线程级，断言也在该线程内执行。
class UtAicpuTsDfxRegOpInfo : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsDfxRegOpInfo tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsDfxRegOpInfo tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsDfxRegOpInfo SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        // 恢复profiling状态、芯片类型与SQE置位开关，防止泄漏污染后续用例
        Hccl::DfxProfilingHandlerLite::GetInstance().SetProL1On(false);
        set_chip_type_stub(0, 0);
        Hccl::SetSqeProfilingEnabled(false);
        GlobalMockObject::verify();
        std::cout << "A Test case in UtAicpuTsDfxRegOpInfo TearDown" << std::endl;
    }

    // 在新线程中执行报到口并断言（绕开主线程thread_local设备类型缓存；g_sqeProfBit断言也在该线程）
    static void RegOpInfoOnFreshThread(bool expectEnabled)
    {
        std::thread worker([expectEnabled]() {
            (void)HcclDfxRegOpInfoByCommId(nullptr, nullptr);
            EXPECT_EQ(Hccl::g_sqeProfBit, expectEnabled ? 1U : 0U);
        });
        worker.join();
    }
};

// L1开启但设备非960：刷新段推送0，SQE置位开关须置0
TEST_F(UtAicpuTsDfxRegOpInfo, Ut_DfxRegOpInfo_L1OnDev950_SqeProfBitStaysZero)
{
    Hccl::DfxProfilingHandlerLite::GetInstance().SetProL1On(true);
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_950));

    RegOpInfoOnFreshThread(false);
}

// L1关闭：不查询设备类型，刷新段推送0
TEST_F(UtAicpuTsDfxRegOpInfo, Ut_DfxRegOpInfo_L1Off_SqeProfBitStaysZero)
{
    Hccl::DfxProfilingHandlerLite::GetInstance().SetProL1On(false);

    (void)HcclDfxRegOpInfoByCommId(nullptr, nullptr);
    EXPECT_EQ(Hccl::g_sqeProfBit, 0U);
}
