/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测对象：src/coll_communicator_mgr/team/hccl/hccl_team_mgr.cc 的 HcclTeamMgr（进程级 team 管理器）：
// 预制 worldTeam 索引（RegisterPrebuiltWorldTeam/FindWorldTeamByProtoLayer/
// GetWorldTeamSizesPerNetLayer/GetRankLayerSlots）、子团队注册（RegisterSubTeam）、索引清理
// （UnregisterTeam/ClearByCollComm）与查询（GetLinkedSubTeams）等（对应 STC ut_ain_021~026/027~029/064）。
// 设计说明：
// 1) HcclTeamMgr 是非虚单例，不可 mockcpp，本文件直接驱动真实单例；
//    TearDown 用 UnregisterTeam/ClearByCollComm 兜底清理，防止单例状态泄漏到下一用例。
// 2) ClearByCollComm 的 ExecuteTeamCleanup 会锁外调用 HcommTeamDestroy/hrtFree（L3/RTS 接口），
//    用例内通过 mockcpp MOCKER(...) 拦截，避免真实调用。
// 3) 通信域采用真实 hcclComm（rank=1、rankSize=2），与 ut_hccl_team_c_adpt.cc 相同的 2P 图构造方式。

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "hccl_team.h"
#include "hccl_team_mgr.h"
#include "hcomm_team_c_adpt.h"
#include "hcomm_result_defs.h"
#include "hccl/hccl_res.h"
#include "hccl_comm_pub.h"
#include "adapter_rts_common.h"
#include "my_rank.h"
#include "llt_hccl_stub_rank_graph.h"
#include "log.h"

#include "../../../hccl_api_base_test.h"
#include "ut_hccl_team_common.h"

using namespace hccl;
using namespace hcomm;

namespace {
HcclResult StubHrtFreeOk(void* devPtr)
{
    (void)devPtr;
    return HCCL_SUCCESS;
}
} // namespace

// ===================== 测试夹具 =====================
class TestHcclTeamMgr : public BaseInit {
public:
    void SetUp() override
    {
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        BaseInit::SetUp();
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        // 先构造真实 hcclComm（InitCollComm 内部调用真实 hrtMalloc 等，必须在其完成后再 mock）。
        hccl_ut::BuildV2HcclComm(hcclCommPtr, rankGraphV2);
        // 注意：此处不预注 HcommTeamDestroy/hrtFree 桩——ClearByCollComm 用例的
        // expects(exactly(3)) 必须先于任何 stubs 注册（mockcpp FIFO，先注册的先命中）。
    }
    void TearDown() override
    {
        // 兜底清理前临时注入 L3/RTS 桩：清理本通信域残留条目（防止真实 HcommTeamDestroy 伪句柄），
        // 清理后 verify() 清零计数，不影响下一用例。
        MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
        MOCKER(hrtFree).stubs().will(invoke(StubHrtFreeOk));
        HcclTeamMgr::GetInstance().ClearByCollComm(hcclCommPtr->GetCollComm());
        GlobalMockObject::verify();
        BaseInit::TearDown();
    }

protected:
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
};

// =====================================================================
// 1. 预制索引（ut_ain_021~023）
// =====================================================================

// ut_ain_021 RegisterPrebuiltWorldTeam 正常登记并建立索引。
TEST_F(TestHcclTeamMgr, Ut_RegisterPrebuiltWorldTeam_When_Valid_Expect_IndexEstablished)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle worldTeam = reinterpret_cast<HcommTeamHandle>(0x10000);
    ASSERT_EQ(
        HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(worldTeam, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2),
        HCCL_SUCCESS);

    EXPECT_EQ(HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(collComm, COMM_PROTOCOL_UB_CTP, 1), worldTeam);

    std::vector<uint32_t> sizes = {9, 9, 9};
    HcclTeamMgr::GetInstance().GetWorldTeamSizesPerNetLayer(collComm, sizes);
    /* 新实现按 maxLayer 动态定长（仅 layer1 注册 → 长度 2），无 layer2 条目 */
    ASSERT_EQ(sizes.size(), 2U);
    EXPECT_EQ(sizes[0], 0U);
    EXPECT_EQ(sizes[1], 2U);

    std::vector<std::pair<uint32_t, uint32_t>> slots;
    HcclTeamMgr::GetInstance().GetRankLayerSlots(collComm, 0, slots);
    ASSERT_EQ(slots.size(), 1U);
    EXPECT_EQ(slots[0].first, 1U);
    EXPECT_EQ(slots[0].second, 0U);

    // worldTeam 排除：GetLinkedSubTeams 为空
    EXPECT_TRUE(HcclTeamMgr::GetInstance().GetLinkedSubTeams(collComm).empty());
}

// ut_ain_022 登记反向：空参 E_PTR；同 handle 重复 E_PARA；(proto,layer) 重复 E_PARA。
TEST_F(TestHcclTeamMgr, Ut_RegisterPrebuiltWorldTeam_When_Invalid_Expect_ReturnPtrOrPara)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle h1 = reinterpret_cast<HcommTeamHandle>(0x11000);
    HcommTeamHandle h2 = reinterpret_cast<HcommTeamHandle>(0x12000);
    auto& mgr = HcclTeamMgr::GetInstance();

    // 空参
    EXPECT_EQ(mgr.RegisterPrebuiltWorldTeam(nullptr, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_E_PTR);
    EXPECT_EQ(mgr.RegisterPrebuiltWorldTeam(h1, nullptr, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_E_PTR);
    EXPECT_EQ(mgr.RegisterPrebuiltWorldTeam(h1, collComm, COMM_PROTOCOL_UB_CTP, 1, nullptr, 2), HCCL_E_PTR);
    EXPECT_TRUE(mgr.GetPrebuiltWorldTeamRanks(nullptr, COMM_PROTOCOL_UB_CTP, 1).empty());

    // 正常登记 1 个
    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(h1, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    // 同 handle 二次登记
    EXPECT_EQ(mgr.RegisterPrebuiltWorldTeam(h1, collComm, COMM_PROTOCOL_UBOE, 2, rankIds, 2), HCCL_E_PARA);
    // 同 (protocol,netLayer) 不同 handle
    EXPECT_EQ(mgr.RegisterPrebuiltWorldTeam(h2, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_E_PARA);
}

// ut_ain_023 同层多协议只记一份 layerRanksMap_。
TEST_F(TestHcclTeamMgr, Ut_RegisterPrebuiltWorldTeam_When_SameLayerMultiProtocol_Expect_LayerRanksOnce)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle h1 = reinterpret_cast<HcommTeamHandle>(0x13000);
    HcommTeamHandle h2 = reinterpret_cast<HcommTeamHandle>(0x14000);
    auto& mgr = HcclTeamMgr::GetInstance();

    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(h1, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(h2, collComm, COMM_PROTOCOL_UBC_TP, 1, rankIds, 2), HCCL_SUCCESS);

    // sizes[1] 只反映一份 rankIds 数量（不翻倍）
    std::vector<uint32_t> sizes = {0, 0, 0};
    mgr.GetWorldTeamSizesPerNetLayer(collComm, sizes);
    EXPECT_EQ(sizes[1], 2U);

    // 每个 rank 每层只 1 个槽位
    std::vector<std::pair<uint32_t, uint32_t>> slots;
    mgr.GetRankLayerSlots(collComm, 0, slots);
    EXPECT_EQ(slots.size(), 1U);
}

// =====================================================================
// 2. 索引清理（ut_ain_024~026）
// =====================================================================

// ut_ain_024 UnregisterTeam 清理预制索引 + 幂等。
TEST_F(TestHcclTeamMgr, Ut_UnregisterTeam_When_PrebuiltWorldTeam_Expect_IndexCleared)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle h1 = reinterpret_cast<HcommTeamHandle>(0x15000);
    auto& mgr = HcclTeamMgr::GetInstance();

    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(h1, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    // 手工注入 syncMemPtr 验证 hrtFree 路径（经真实 RegisterSubTeam 无法给 worldTeam 挂 syncMem）
    mgr.UnregisterTeam(h1);
    EXPECT_EQ(mgr.FindWorldTeamByProtoLayer(collComm, COMM_PROTOCOL_UB_CTP, 1), nullptr);
    // 重复 UnregisterTeam 不崩溃
    mgr.UnregisterTeam(h1);
}

// ut_ain_025/026 ClearByCollComm 清理指定通信域全部条目（worldTeamIndex_ + layerRanksMap_ 同步清理），
// 其他通信域条目保留。
TEST_F(TestHcclTeamMgr, Ut_ClearByCollComm_When_MixedComms_Expect_OnlyOwnEntriesCleared)
{
    CollComm* collCommA = hcclCommPtr->GetCollComm();
    ASSERT_NE(collCommA, nullptr);
    CollComm* collCommB = reinterpret_cast<CollComm*>(0x90000); // 伪造另一通信域（仅作 map 值）
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle wA1 = reinterpret_cast<HcommTeamHandle>(0xA1000);
    HcommTeamHandle wA2 = reinterpret_cast<HcommTeamHandle>(0xA2000);
    HcommTeamHandle wB1 = reinterpret_cast<HcommTeamHandle>(0xB1000);
    auto& mgr = HcclTeamMgr::GetInstance();

    // comm A：2 个 worldTeam（不同层）+ 1 个 subTeam
    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(wA1, collCommA, COMM_PROTOCOL_UB_CTP, 0, rankIds, 2), HCCL_SUCCESS);
    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(wA2, collCommA, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    HcommTeamHandle subA = reinterpret_cast<HcommTeamHandle>(0xA3000);
    ASSERT_EQ(mgr.RegisterSubTeam(wA1, subA, nullptr, 0, rankIds, 2), HCCL_SUCCESS);
    // comm B：1 个 worldTeam（改用不同 protocol 前缀，避免与残留索引键冲突）
    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(wB1, collCommB, COMM_PROTOCOL_UBOE, 2, rankIds, 2), HCCL_SUCCESS);

    // 销毁断言：ClearByCollComm 锁外销毁 3 条目，各调 HcommTeamDestroy 1 次；hrtFree 空转（syncMem 全空）
    MOCKER(HcommTeamDestroy).expects(exactly(3)).will(returnValue(0));
    MOCKER(hrtFree).stubs().will(invoke(StubHrtFreeOk));

    mgr.ClearByCollComm(collCommA);
    // A 的条目全部清理
    EXPECT_EQ(mgr.FindWorldTeamByProtoLayer(collCommA, COMM_PROTOCOL_UB_CTP, 0), nullptr);
    EXPECT_EQ(mgr.FindWorldTeamByProtoLayer(collCommA, COMM_PROTOCOL_UB_CTP, 1), nullptr);
    EXPECT_EQ(mgr.FindCollComm(subA), nullptr);
    // layerRanksMap_ 按通信域清理：A 的 layer0/layer1 已清，rank0 仅剩 B 的 layer2 槽位
    std::vector<std::pair<uint32_t, uint32_t>> slotsA;
    mgr.GetRankLayerSlots(collCommB, 0, slotsA);
    ASSERT_EQ(slotsA.size(), 1U);
    EXPECT_EQ(slotsA[0].first, 2U); // 仅 B 的 layer2
    // B 的条目保留
    EXPECT_EQ(mgr.FindWorldTeamByProtoLayer(collCommB, COMM_PROTOCOL_UBOE, 2), wB1);
    mgr.UnregisterTeam(wB1); // 清理 B
}

// =====================================================================
// 3. RegisterSubTeam 与 GetLinkedSubTeams（ut_ain_027~029）
// =====================================================================

// ut_ain_027 RegisterSubTeam 正常 + protocol 继承 + collComm 继承。
TEST_F(TestHcclTeamMgr, Ut_RegisterSubTeam_When_Valid_Expect_ProtocolInherited)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle worldTeam = reinterpret_cast<HcommTeamHandle>(0x16000);
    HcommTeamHandle subTeam = reinterpret_cast<HcommTeamHandle>(0x17000);
    auto& mgr = HcclTeamMgr::GetInstance();

    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(worldTeam, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    // syncMemPtr 传 nullptr（本用例不校验 syncMem；非空伪指针会在 TearDown 兜底 UnregisterTeam 时
    // 走真实 hrtFree 触发 SEGFAULT）
    ASSERT_EQ(mgr.RegisterSubTeam(worldTeam, subTeam, nullptr, 0, rankIds, 2), HCCL_SUCCESS);

    // FindWorldTeam(subTeam)==worldTeam；rankIds 正确；collComm 继承
    EXPECT_EQ(mgr.FindWorldTeam(subTeam), worldTeam);
    auto subRanks = mgr.GetRankIds(subTeam);
    EXPECT_EQ(subRanks.size(), 2U);
    EXPECT_EQ(mgr.FindCollComm(subTeam), collComm);
}

// ut_ain_028 RegisterSubTeam 反向：空参 E_PTR；worldTeam 未注册 E_PARA；subTeam 重复 E_PARA。
TEST_F(TestHcclTeamMgr, Ut_RegisterSubTeam_When_Invalid_Expect_ReturnPtrOrPara)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle worldTeam = reinterpret_cast<HcommTeamHandle>(0x18000);
    HcommTeamHandle subTeam = reinterpret_cast<HcommTeamHandle>(0x19000);
    HcommTeamHandle subTeam2 = reinterpret_cast<HcommTeamHandle>(0x1A000);
    auto& mgr = HcclTeamMgr::GetInstance();

    // 空参
    EXPECT_EQ(mgr.RegisterSubTeam(nullptr, subTeam, nullptr, 0, rankIds, 2), HCCL_E_PTR);
    EXPECT_EQ(mgr.RegisterSubTeam(worldTeam, nullptr, nullptr, 0, rankIds, 2), HCCL_E_PTR);
    EXPECT_EQ(mgr.RegisterSubTeam(worldTeam, subTeam, nullptr, 0, nullptr, 2), HCCL_E_PTR);
    // worldTeam 未注册
    EXPECT_EQ(mgr.RegisterSubTeam(worldTeam, subTeam, nullptr, 0, rankIds, 2), HCCL_E_PARA);

    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(worldTeam, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    ASSERT_EQ(mgr.RegisterSubTeam(worldTeam, subTeam, nullptr, 0, rankIds, 2), HCCL_SUCCESS);
    // subTeam 重复
    EXPECT_EQ(mgr.RegisterSubTeam(worldTeam, subTeam, nullptr, 0, rankIds, 2), HCCL_E_PARA);
    (void)subTeam2;
}

// ut_ain_029 GetLinkedSubTeams 过滤：仅本 comm 的 subTeam，worldTeam 排除；空 comm 返回空。
TEST_F(TestHcclTeamMgr, Ut_GetLinkedSubTeams_When_Mixed_Expect_OnlyOwnSubTeams)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle worldTeam = reinterpret_cast<HcommTeamHandle>(0x1B000);
    HcommTeamHandle sub1 = reinterpret_cast<HcommTeamHandle>(0x1C000);
    HcommTeamHandle sub2 = reinterpret_cast<HcommTeamHandle>(0x1D000);
    CollComm* collCommB = reinterpret_cast<CollComm*>(0x91000);
    HcommTeamHandle worldB = reinterpret_cast<HcommTeamHandle>(0x1E000);
    HcommTeamHandle subB = reinterpret_cast<HcommTeamHandle>(0x1F000);
    auto& mgr = HcclTeamMgr::GetInstance();

    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(worldTeam, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    ASSERT_EQ(mgr.RegisterSubTeam(worldTeam, sub1, nullptr, 0, rankIds, 2), HCCL_SUCCESS);
    ASSERT_EQ(mgr.RegisterSubTeam(worldTeam, sub2, nullptr, 0, rankIds, 2), HCCL_SUCCESS);
    // comm B 的 worldTeam 用不同 protocol（避免与可能残留的同 (proto,layer) 索引键冲突）
    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(worldB, collCommB, COMM_PROTOCOL_UBOE, 2, rankIds, 2), HCCL_SUCCESS);
    ASSERT_EQ(mgr.RegisterSubTeam(worldB, subB, nullptr, 0, rankIds, 2), HCCL_SUCCESS);

    auto subs = mgr.GetLinkedSubTeams(collComm);
    ASSERT_EQ(subs.size(), 2U);
    EXPECT_TRUE(std::find(subs.begin(), subs.end(), sub1) != subs.end());
    EXPECT_TRUE(std::find(subs.begin(), subs.end(), sub2) != subs.end());
    EXPECT_TRUE(std::find(subs.begin(), subs.end(), worldTeam) == subs.end()); // worldTeam 排除

    EXPECT_TRUE(mgr.GetLinkedSubTeams(nullptr).empty()); // 空 comm
}

// =====================================================================
// 4. CollectPendingMemHandles（ut_ain_064）
// =====================================================================

// ut_ain_064 CollectPendingMemHandles 只收集一次（syncMemExchanged 置位）。
TEST_F(TestHcclTeamMgr, Ut_CollectPendingMemHandles_When_CalledTwice_Expect_CollectOnce)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t rankIds[2] = {0, 1};
    HcclMemHandle fakeUserMemHandle = reinterpret_cast<HcclMemHandle>(0x60000);
    HcommTeamHandle worldTeam = reinterpret_cast<HcommTeamHandle>(0x21000);
    HcommTeamHandle subTeam = reinterpret_cast<HcommTeamHandle>(0x22000);
    auto& mgr = HcclTeamMgr::GetInstance();

    ASSERT_EQ(mgr.RegisterPrebuiltWorldTeam(worldTeam, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2), HCCL_SUCCESS);
    ASSERT_EQ(mgr.RegisterSubTeam(worldTeam, subTeam, nullptr, 0, rankIds, 2), HCCL_SUCCESS);
    mgr.SetTeamSyncMemHandle(subTeam, fakeUserMemHandle, "__hccl_team_syncmem__ut");

    auto handles1 = mgr.CollectPendingMemHandles(worldTeam, subTeam);
    EXPECT_EQ(handles1.size(), 1U);
    EXPECT_EQ(handles1[0], fakeUserMemHandle);
    // 第二次：syncMemExchanged 已置位，返回空
    auto handles2 = mgr.CollectPendingMemHandles(worldTeam, subTeam);
    EXPECT_TRUE(handles2.empty());
}
