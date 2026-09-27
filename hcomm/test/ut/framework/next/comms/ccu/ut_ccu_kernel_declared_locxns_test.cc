/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 目的:
//   验证 CcuKernel::declaredLocXns_ 的"精细化 pinned Xn"登记契约 --
//   只有算法侧显式调用 VariableCreateByChannel (即 GetResByChannel<Variable>) 触及
//   的 (channel, varIndex) 才会被视为对端 SyncWtX 的可能落点, 登记进
//   GetDeclaredLocXns(); 未声明的 locXn 槽位不会被 pin, 允许 register reallocator
//   自由重命名. 这份契约取代此前"整包 pin channel locRes_.xns"的保守做法.

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "hcomm_c_adpt.h"

#define private public
#define protected public
#include "ccu_kernel.h"
#include "ccu_urma_channel.h"
#undef private
#undef protected

using namespace hcomm;

class CcuKernelDeclaredLocXnsTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

// 场景 1: 未做任何 VariableCreateByChannel 调用时, 集合应为空.
// (对应"纯生产者 kernel"或空 kernel: 不应 pin 任何本 rank locXn.)
TEST_F(CcuKernelDeclaredLocXnsTest, Ut_Empty_When_NoCreateVariable_Expect_EmptySet)
{
    hcomm::CcuKernel kernel;
    EXPECT_TRUE(kernel.GetDeclaredLocXns().empty());
}

// 场景 2: 单 channel + 2 个不同 varIndex -> 精确 2 个物理 xn 登记进集合.
// 关键点: 8 元素 locRes_.xns 池里, 只有被声明的两个索引对应的物理号被 pin.
TEST_F(CcuKernelDeclaredLocXnsTest, Ut_SingleChannel_TwoDistinctVarIndex_Expect_ExactlyTwoPinnedXns)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0xAAA;

    // outBound(v) 会匹配任意实参, 并把 v 的当前值写回该 out 参数.
    uint32_t out101 = 101;
    uint32_t out102 = 102;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(eq(handle), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    // varIndex=3 -> 物理 xn=101; varIndex=5 -> 物理 xn=102 (其余 6 个未声明).
    MOCKER_CPP(&CcuUrmaChannel::GetLocXnByIndex)
        .stubs()
        .with(eq(static_cast<uint32_t>(3)), outBound(out101))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetLocXnByIndex)
        .stubs()
        .with(eq(static_cast<uint32_t>(5)), outBound(out102))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetDieId).stubs().will(returnValue(static_cast<uint32_t>(0)));

    hcomm::CcuKernel kernel;
    CcuVariableHandle vh1 = 0;
    CcuVariableHandle vh2 = 0;
    ASSERT_EQ(kernel.VariableCreateByChannel(handle, 3, &vh1), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(kernel.VariableCreateByChannel(handle, 5, &vh2), CcuResult::CCU_SUCCESS);

    const auto& decl = kernel.GetDeclaredLocXns();
    EXPECT_EQ(decl.size(), 2u);
    EXPECT_TRUE(decl.count(101U));
    EXPECT_TRUE(decl.count(102U));
    // 关键回归点: 集合规模 << channel locRes_.xns 池规模 (INIT_XN_NUM=8).
    EXPECT_LT(decl.size(), 8u);
}

// 场景 3: 同一 (channel, varIndex) 声明多次, declaredLocXns_ 应按物理号自动去重.
TEST_F(CcuKernelDeclaredLocXnsTest, Ut_SameVarIndexDeclaredTwice_Expect_PhysicalIdDeduplicated)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0xBBB;

    uint32_t out200 = 200;
    MOCKER(HcommChannelGet)
        .stubs()
        .with(eq(handle), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    MOCKER_CPP(&CcuUrmaChannel::GetLocXnByIndex)
        .stubs()
        .with(eq(static_cast<uint32_t>(2)), outBound(out200))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetDieId).stubs().will(returnValue(static_cast<uint32_t>(0)));

    hcomm::CcuKernel kernel;
    CcuVariableHandle vh1 = 0;
    CcuVariableHandle vh2 = 0;
    ASSERT_EQ(kernel.VariableCreateByChannel(handle, 2, &vh1), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(kernel.VariableCreateByChannel(handle, 2, &vh2), CcuResult::CCU_SUCCESS);
    // Variable handle 侧允许重复 alloc (拿到不同的 CcuVariableHandle), 但底层
    // 物理 xn 号相同 -> declaredLocXns_ 只保留一份.
    EXPECT_EQ(kernel.GetDeclaredLocXns().size(), 1u);
    EXPECT_TRUE(kernel.GetDeclaredLocXns().count(200U));
}

// 场景 4: 两条不同 channel 分别声明各自的 xn, 集合应含两条 channel 的物理号.
// 用两个不同的 varIndex (0 vs 1) 区分, 避免同一 varIndex 下 outBound 出现两次
// 无法按 first-arg 消歧的问题.
TEST_F(CcuKernelDeclaredLocXnsTest, Ut_TwoChannels_SeparateDeclarations_Expect_UnionOfPhysicalIds)
{
    HcommChannelDesc desc1{};
    HcommChannelDesc desc2{};
    CcuUrmaChannel channel1(nullptr, desc1);
    CcuUrmaChannel channel2(nullptr, desc2);
    void* channelPtr1 = static_cast<Channel*>(&channel1);
    void* channelPtr2 = static_cast<Channel*>(&channel2);
    ChannelHandle handle1 = 0x100;
    ChannelHandle handle2 = 0x200;

    uint32_t out10 = 10;
    uint32_t out20 = 20;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(eq(handle1), outBoundP(&channelPtr1))
        .will(returnValue(static_cast<HcommResult>(0)));
    MOCKER(HcommChannelGet)
        .stubs()
        .with(eq(handle2), outBoundP(&channelPtr2))
        .will(returnValue(static_cast<HcommResult>(0)));
    // channel1 声明 varIndex=0 -> 物理 xn=10; channel2 声明 varIndex=1 -> 物理 xn=20.
    MOCKER_CPP(&CcuUrmaChannel::GetLocXnByIndex)
        .stubs()
        .with(eq(static_cast<uint32_t>(0)), outBound(out10))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetLocXnByIndex)
        .stubs()
        .with(eq(static_cast<uint32_t>(1)), outBound(out20))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuUrmaChannel::GetDieId).stubs().will(returnValue(static_cast<uint32_t>(0)));

    hcomm::CcuKernel kernel;
    CcuVariableHandle vh1 = 0;
    CcuVariableHandle vh2 = 0;
    ASSERT_EQ(kernel.VariableCreateByChannel(handle1, 0, &vh1), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(kernel.VariableCreateByChannel(handle2, 1, &vh2), CcuResult::CCU_SUCCESS);

    const auto& decl = kernel.GetDeclaredLocXns();
    EXPECT_EQ(decl.size(), 2u);
    EXPECT_TRUE(decl.count(10U));
    EXPECT_TRUE(decl.count(20U));
}

// 场景 5: GetLocXnByIndex 失败 -> 不应登记 (契约: 仅在成功查到物理号后才写入集合).
TEST_F(CcuKernelDeclaredLocXnsTest, Ut_GetLocXnByIndexFails_Expect_NotRegistered)
{
    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);
    ChannelHandle handle = 0xCCC;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(eq(handle), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    MOCKER_CPP(&CcuUrmaChannel::GetLocXnByIndex).stubs().will(returnValue(HcclResult::HCCL_E_PARA));

    hcomm::CcuKernel kernel;
    CcuVariableHandle vh = 0;
    // 失败路径: 期望上层 CHK_RET 抛错; declaredLocXns_ 保持空.
    (void)kernel.VariableCreateByChannel(handle, 3, &vh);
    EXPECT_TRUE(kernel.GetDeclaredLocXns().empty());
}
