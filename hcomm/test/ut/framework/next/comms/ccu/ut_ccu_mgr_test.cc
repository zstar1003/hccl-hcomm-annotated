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

#include "hccl_ccu_res.h"
#include "hccl_types.h"
#include "ccu_kernel.h"

#include "src/base_comm/resources/ccu/ccu_instance/ccu_res_pack.h"
#include "ccu_kernel_mgr.h"
#include "ccu_res_repo.h"
#include "ccu_common.h"

#undef private
#undef protected

using namespace hcomm;

class CcuMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuMgrTest tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CcuMgrTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuMgrTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuMgrTest TearDown" << std::endl;
    }
};

class MockCcuKernelArg : public hcomm::CcuKernelArg {
public:
    CcuKernelSignature GetKernelSignature() const override { return CcuKernelSignature{}; }
};

class MockCcuKernel : public hcomm::CcuKernel {
public:
    explicit MockCcuKernel(const hcomm::CcuKernelArg& arg) : hcomm::CcuKernel(arg) {}

    HcclResult Algorithm() override { return HCCL_SUCCESS; }

    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override { return {}; }
};

TEST_F(CcuMgrTest, Ut_CcuRegister_When_resourceNotEnough_Expect_Return_false)
{
    MOCKER_CPP(&hcomm::CcuKernelMgr::Init).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::CcuKernel::Init).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    // kernel 请求的资源：ms[0] = 1，这样 CheckResIfAvailable 会发现资源不足
    hcomm::CcuResReq resReq{};
    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        resReq.msReq[i] = 1;
    }
    MOCKER_CPP(&hcomm::CcuKernel::GetResourceRequest).stubs().will(returnValue(resReq));

    // Mock AllocInstrRes 让它成功返回，这样能走到 CheckResIfAvailable
    MOCKER_CPP(&hcomm::CcuDevMgrImp::AllocIns).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    uint32_t devicePhyId = 0;
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(devicePhyId);
    MockCcuKernelArg kernelArg;

    // 给 KernelCreator 赋值一个有效的 lambda
    hcomm::KernelCreator creator = [](const hcomm::CcuKernelArg& arg) {
        return std::make_unique<MockCcuKernel>(arg);
    };
    const auto& arg = *static_cast<const hcomm::CcuKernelArg*>(&kernelArg);
    std::unique_ptr<hcomm::CcuKernel> kernel = creator(arg);

    // 构造一个空资源的 resPack（已持有资源为0）
    // 注意：CcuResPack 是 explicit 构造，需要传入 CcuEngine
    hcomm::CcuResPack* resPack = new hcomm::CcuResPack(hcomm::CcuEngine::CCU_MS);
    // 这里 resPack 的 resRepo_ 初始是空的，所以 GetResNumFromResPack 得到的 totalRes 全是0
    // 而 kernel 请求了 msReq[0..N] = 1，所以 CheckResIfAvailable 会返回 false

    CcuKernelHandle newHandle{0};
    HcclResult ret = kernelMgr.Register(std::move(kernel), *resPack, newHandle);
    EXPECT_EQ(ret, HcclResult::HCCL_E_UNAVAIL);
}
