/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "hccl_mem_defs.h"
#define private public
#define protected public
#include "global_mem_record.h"
#include "global_mem_manager.h"
#undef protected
#undef private

using namespace std;
using namespace hccl;

class GlobalMemMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--GlobalMemMgrTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--GlobalMemMgrTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(GlobalMemMgrTest, ut_global_mem_mgr_reg)
{
    GlobalMemRegMgr mgr;
    HcclResult ret = HCCL_SUCCESS;

    auto buffer1 = std::vector<int8_t>(10);

    // 不允许注册空内存
    void* nullHandle = nullptr;
    HcclMem nullMem{HCCL_MEM_TYPE_DEVICE, nullptr, buffer1.size()};
    ret = mgr.Reg(&nullMem, &nullHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
    HcclMem zeroMem{HCCL_MEM_TYPE_DEVICE, buffer1.data(), 0};
    ret = mgr.Reg(&zeroMem, &nullHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclMem mem1{HCCL_MEM_TYPE_DEVICE, buffer1.data(), buffer1.size()};
    void* memHandle1 = nullptr;
    ret = mgr.Reg(&mem1, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto buffer2 = std::vector<int8_t>(10);
    HcclMem mem2{HCCL_MEM_TYPE_DEVICE, buffer2.data(), buffer2.size()};
    void* memHandle2 = nullptr;
    ret = mgr.Reg(&mem2, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 注册不同类型，但是地址一样
    void* memHandle3 = nullptr;
    HcclMem mem3 = {HCCL_MEM_TYPE_HOST, buffer2.data(), buffer2.size()};
    ret = mgr.Reg(&mem3, &memHandle3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 不允许注册交集
    HcclMem mem4 = {HCCL_MEM_TYPE_HOST, buffer2.data(), buffer2.size()};
    mem4.addr = buffer2.data() + 1;
    mem4.size = buffer2.size() - 1;
    ret = mgr.Reg(&mem4, &nullHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
    mem4.addr = buffer2.data() - 1;
    mem4.size = buffer2.size() + 1;
    ret = mgr.Reg(&mem4, &nullHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(GlobalMemMgrTest, ut_global_mem_mgr_dereg)
{
    GlobalMemRegMgr mgr;
    HcclResult ret = HCCL_SUCCESS;

    auto buffer1 = std::vector<int8_t>(10);
    HcclMem mem1{HCCL_MEM_TYPE_DEVICE, buffer1.data(), buffer1.size()};

    // 没注册就析构报Not found
    {
        GlobalMemRecord record(mem1);
        ret = mgr.DeReg(&record);
        EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    }

    void* memHandle1 = nullptr;
    ret = mgr.Reg(&mem1, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto buffer2 = std::vector<int8_t>(10);
    HcclMem mem2{HCCL_MEM_TYPE_DEVICE, buffer2.data(), buffer2.size()};
    void* memHandle2 = nullptr;
    ret = mgr.Reg(&mem2, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = mgr.DeReg(memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 绑定着通信域的内存不能直接注销
    auto recordPtr = reinterpret_cast<GlobalMemRecord*>(memHandle2);
    ret = recordPtr->BindToComm("comm1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = mgr.DeReg(memHandle2);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = recordPtr->UnbindFromComm("comm1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = mgr.DeReg(memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult hrtGetPairDevicePhyIdForTest(u32 localDevPhyId, u32& pairDevPhyId)
{
    pairDevPhyId = 1;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceTypeForTest(DevType& devType)
{
    devType = DevType::DEV_TYPE_910_93;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceIndexByPhyIdForTest(u32 devicePhyId, u32& deviceLogicId)
{
    deviceLogicId = 1;
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDeviceAllNicIPForTest(std::vector<std::vector<HcclIpAddress>>& ipAddr)
{
    ipAddr.clear();
    HcclIpAddress testIp1{"10.10.10.11"};
    std::vector<HcclIpAddress> vec1;
    vec1.push_back(testIp1);
    HcclIpAddress testIp2{"10.10.10.12"};
    std::vector<HcclIpAddress> vec2;
    vec2.push_back(testIp2);
    ipAddr.push_back(vec1);
    ipAddr.push_back(vec2);
    GTEST_LOG_(INFO) << "lyy ipAddr.size: " << ipAddr.size();

    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDeviceIPForTest(u32 devicePhyId, std::vector<hccl::HcclIpAddress>& ipAddr)
{
    ipAddr.clear();
    hccl::HcclIpAddress testIp1{"10.10.10.11"};
    ipAddr.push_back(testIp1);
    return HCCL_SUCCESS;
}

TEST_F(GlobalMemMgrTest, ut_global_mem_mgr_backupInit)
{
    MOCKER(hrtGetPairDevicePhyId).stubs().will(invoke(hrtGetPairDevicePhyIdForTest));
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().will(invoke(hrtGetDeviceIndexByPhyIdForTest));
    MOCKER(hrtGetDeviceType).stubs().will(invoke(hrtGetDeviceTypeForTest));
    MOCKER(hrtRaGetDeviceIP).stubs().will(invoke(hrtRaGetDeviceIPForTest));
    MOCKER(hrtRaGetDeviceAllNicIP).stubs().will(invoke(hrtRaGetDeviceAllNicIPForTest));
    MOCKER(HcclNetInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::ServerInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::ServerDeInit, HcclResult(HcclSocketManager::*)(const HcclNetDevCtx, u32))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclNetDevCtx ctx;
    HcclIpAddress remoteIp1{"10.10.10.11"};
    HcclIpAddress remoteIp2{"10.10.10.12"};
    u32 port = 16667;

    GlobalMemRegMgr mgr;
    HcclResult ret = mgr.GetNetDevCtx(NicType::DEVICE_NIC_TYPE, remoteIp1, port, ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

HcclResult hrtGetDeviceRefreshForTest(s32* deviceLogicID)
{
    *deviceLogicID = 0;
    return HCCL_SUCCESS;
}

TEST_F(GlobalMemMgrTest, Ut_GlobalMemMgr_InicNic_When_Normal_Expect_Success)
{
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(hrtGetDeviceRefreshForTest));
    MOCKER(HcclNetInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDeInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = GlobalMemRegMgr::GetInstance().InitNic();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = GlobalMemRegMgr::GetInstance().DeInitNic();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}
