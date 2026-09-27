/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测对象：src/coll_communicator_mgr/team/hccl/hccl_team_c_adpt.cc 的 HcclTeamCreate/HcclTeamDestroy
// （含该文件内静态工具函数的分支覆盖），以及 HcclCommMemReg 前缀校验（对应 STC ut_ain_001~020/063/065~067）。
// HcclTeamMgr（hccl_team_mgr.cc）的用例见同目录 ut_hccl_team_mgr.cc。
// 设计说明：
// 1) Hcomm 层 team 接口（HcommTeamCreate/Destroy/GetNetLayer/BindChannels/BindRemoteSyncMem）编译入 hccl_llt
//    共享库；用例内通过 mockcpp MOCKER(...) 拦截并按场景设置返回值/出参。
// 2) HcclTeamMgr 是非虚单例，不可 mockcpp。worldTeam 预制通过 RegisterPrebuiltWorldTeam 真实登记，
//    subTeam 经被测 HcclTeamCreate 真实注册，TearDown 用 UnregisterTeam 兜底清理防止单例状态泄漏。
// 3) 通信域采用真实 hcclComm + RankGraphStub 2P 图：rank=1、rankSize=2，对端 rank=0。
// 4) CommMems::CommRegMem/CommUnregMem 为非虚成员函数，用 MOCKER_CPP 拦截（hccl_llt 已含实现）。

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"

#include <cstring>
#include <securec.h>
#include <string>
#include <vector>

#include "hccl_team.h"
#include "hccl_team_c_adpt.h"
#include "hcomm_team_c_adpt.h"
#include "hcomm_team_mgr.h"
#include "hcomm_result_defs.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_types.h"
#include "hccl_comm_pub.h"
#include "adapter_rts_common.h"
#include "hccl_team_mgr.h"
#include "comm_mems.h"
#include "my_rank.h"
#include "llt_hccl_stub_rank_graph.h"
#include "log.h"
#include "topoinfo_struct.h"

#include "../../../hccl_api_base_test.h"
#include "ut_hccl_team_common.h"

using namespace hccl;
using namespace hcomm;

namespace {
// ===================== 公共测试常量 =====================
// worldTeam/subTeam 的伪造句柄值（非空即可，仅作为 map key 与 mock 出参）。
HcommTeamHandle g_fakeWorldTeam = reinterpret_cast<HcommTeamHandle>(0x10000);
HcommTeamHandle g_fakeSubTeam = reinterpret_cast<HcommTeamHandle>(0x20000);
void* g_fakeWsPtr = reinterpret_cast<void*>(0x40000);
HcclMemHandle g_fakeUserMemHandle = reinterpret_cast<HcclMemHandle>(0x60000);
ChannelHandle g_fakeChannel = static_cast<ChannelHandle>(0x70000);

// ===================== mockcpp invoke 桩 =====================
// HcommTeamCreate 成功桩：返回 g_fakeSubTeam + syncMemSize=4096。
HcommResult StubSubTeamCreateOk(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = g_fakeSubTeam;
    *outSyncMemSize = 4096;
    return 0;
}
// HcommTeamCreate 失败桩：返回错误码。
HcommResult StubSubTeamCreateFail(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = nullptr;
    *outSyncMemSize = 0;
    return static_cast<HcommResult>(HCOMM_E_INTERNAL);
}
// HcommTeamCreate 返回 ok 但 *team=nullptr（触发 HCCL_E_INTERNAL 分支）。
HcommResult StubSubTeamCreateNullTeam(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = nullptr;
    *outSyncMemSize = 0;
    return 0;
}
// HcommTeamCreate 成功但 syncMemSize=0（触发 HCCL_E_PARA 分支，源码不回滚）。
HcommResult StubSubTeamCreateZeroWs(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = g_fakeSubTeam;
    *outSyncMemSize = 0;
    return 0;
}
HcommResult StubTeamGetNetLayerOne(HcommTeamHandle team, uint32_t* netLayer)
{
    (void)team;
    *netLayer = 1;
    return 0;
}

// hrtMalloc 出参桩：填入伪指针。
HcclResult StubHrtMallocFill(void** devPtr, u64 size, bool level2Address)
{
    (void)size;
    (void)level2Address;
    *devPtr = g_fakeWsPtr;
    return HCCL_SUCCESS;
}

// —— hrtMalloc 计数桩：第 1 次（worldTeam L3 资源外）失败 ——
uint32_t g_hrtMallocOkCount = 0;
uint32_t g_hrtMallocCallCnt = 0;
HcclResult StubHrtMallocCountFail(void** devPtr, u64 size, bool level2Address)
{
    (void)size;
    (void)level2Address;
    g_hrtMallocCallCnt++;
    if (g_hrtMallocCallCnt <= g_hrtMallocOkCount) {
        *devPtr = g_fakeWsPtr;
        return HCCL_SUCCESS;
    }
    *devPtr = nullptr;
    return HCCL_E_MEMORY;
}

// HcclRankGraphGetLinks 出参桩：返回单条 UB_CTP 链路。
CommLink g_fakeLink{};
HcclResult StubRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    (void)comm;
    (void)netLayer;
    (void)srcRank;
    (void)dstRank;
    g_fakeLink.header.version = 1;
    g_fakeLink.header.magicWord = 0x0f0e0f0f;
    g_fakeLink.header.size = sizeof(CommLink);
    g_fakeLink.header.reserved = 0;
    g_fakeLink.linkAttr.linkProtocol = COMM_PROTOCOL_UB_CTP;
    g_fakeLink.linkAttr.hop = 1;
    *links = &g_fakeLink;
    *linkNum = 1;
    return HCCL_SUCCESS;
}
// HcclRankGraphGetLinks 桩：无 link（linkNum=0）→ 后续 E_NOT_FOUND。
HcclResult StubRankGraphGetLinksEmpty(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    (void)comm;
    (void)netLayer;
    (void)srcRank;
    (void)dstRank;
    *links = nullptr;
    *linkNum = 0;
    return HCCL_SUCCESS;
}

// HcclChannelAcquire 出参桩：每个 channel 填入伪造句柄。
HcclResult StubChannelAcquireFill(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    (void)comm;
    (void)engine;
    (void)channelDescs;
    for (uint32_t i = 0; i < channelNum; i++) {
        channels[i] = g_fakeChannel + i;
    }
    return HCCL_SUCCESS;
}

// HcclChannelGetRemoteMems 出参桩：返回带 __hccl_team_syncmem__ 前缀 tag 的远端内存。
CommMem g_remoteMemsBuf[2];
char* g_remoteTagsBuf[2];
char g_syncTagBuf[128];
HcclResult StubChannelGetRemoteMemsSync(
    HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags)
{
    (void)comm;
    (void)channel;
    g_remoteMemsBuf[0].type = COMM_MEM_TYPE_DEVICE;
    g_remoteMemsBuf[0].addr = reinterpret_cast<void*>(0x41000);
    g_remoteMemsBuf[0].size = 1024;
    if (strcpy_s(g_syncTagBuf, sizeof(g_syncTagBuf), "__hccl_team_syncmem__ut") != EOK) {
        return HCCL_E_INTERNAL;
    }
    g_remoteTagsBuf[0] = g_syncTagBuf;
    *memNum = 1;
    *remoteMems = g_remoteMemsBuf;
    *memTags = g_remoteTagsBuf;
    return HCCL_SUCCESS;
}

// CommMems::CommRegMem/CommUnregMem 桩（非虚成员函数，MOCKER_CPP 拦截）。
HcclResult StubCommRegMemOk(CommMems* self, const std::string& tag, const CommMem& mem, void** handle)
{
    (void)self;
    (void)tag;
    (void)mem;
    *handle = g_fakeUserMemHandle;
    return HCCL_SUCCESS;
}
// CommUnregMem 可配置返回值。
HcclResult g_commUnregMemRet = HCCL_SUCCESS;
uint32_t g_commUnregMemCallCnt = 0;
HcclResult StubCommUnregMem(CommMems* self, const std::string& tag, const void* handle)
{
    (void)self;
    (void)tag;
    (void)handle;
    g_commUnregMemCallCnt++;
    return g_commUnregMemRet;
}
// CommRegMem 计数桩（幂等场景验证 CommRegMem 不被调用）。
uint32_t g_commRegMemCallCnt = 0;
HcclResult StubCommRegMemCountFail(CommMems* self, const std::string& tag, const CommMem& mem, void** handle)
{
    (void)self;
    (void)tag;
    (void)mem;
    (void)handle;
    g_commRegMemCallCnt++;
    return HCCL_SUCCESS;
}

// ===================== 公共辅助 =====================
// BuildV2HcclComm 见 ut_hccl_team_common.h（与 ut_hccl_team_mgr.cc 共享）。

// 构造并初始化 HcclTeamCreateDesc（合法默认值：rankIds={0,1}, rankNum=2, selfRankId=1, netLayer=1, UB_CTP, AIV）。
void BuildTeamCreateDesc(
    HcclTeamCreateDesc& desc, const uint32_t* rankIds, uint32_t rankNum, uint32_t selfRankId, uint32_t netLayer = 1)
{
    (void)HcclTeamCreateDescInit(&desc);
    desc.rankIds = rankIds;
    desc.rankNum = rankNum;
    desc.selfRankId = selfRankId;
    desc.netLayer = netLayer;
    desc.protocol = COMM_PROTOCOL_UB_CTP;
    desc.engine = COMM_ENGINE_AIV;
    desc.requirement.signalCount = 0;
    desc.requirement.counterCount = 0;
    desc.requirement.barrierCount = 1;
    desc.channelCnt = 1;
    desc.notifyNum = 8;
}
} // namespace

// ===================== 测试夹具 =====================
class TestCollCommTeamCAdpt : public BaseInit {
public:
    void SetUp() override
    {
        // main.cc 设置的 connection_fault_detction_time:0 已不被 EnvConfig 支持，单用例运行时
        // （按 gtest_filter 过滤，无 AivUbMemTransportTest 先行覆盖）会抛 Invalid params 异常，
        // 此处覆盖为合法值（与 ut_AivUbMemTransport_API_test.cc SetUp 相同做法）。
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        BaseInit::SetUp();
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        // 先构造真实 hcclComm（InitCollComm 内部调用真实 hrtMalloc 等，必须在其完成后再 mock）。
        hccl_ut::BuildV2HcclComm(hcclCommPtr, rankGraphV2);
        comm = static_cast<HcclComm>(hcclCommPtr.get());
        g_commUnregMemRet = HCCL_SUCCESS;
        g_commUnregMemCallCnt = 0;
        g_hrtMallocOkCount = 0;
        g_hrtMallocCallCnt = 0;
    }
    void TearDown() override
    {
        // 兜底清理：直接 UnregisterTeam 清除可能残留的 team 条目，防止单例状态泄漏到下一用例。
        HcclTeamMgr::GetInstance().UnregisterTeam(g_fakeWorldTeam);
        HcclTeamMgr::GetInstance().UnregisterTeam(g_fakeSubTeam);
        GlobalMockObject::verify();
        BaseInit::TearDown();
    }

protected:
    // 真实登记一个预制 worldTeam（protocol=UB_CTP, netLayer=1, rankIds={0,1}）。
    void RegisterDefaultWorldTeam()
    {
        uint32_t rankIds[2] = {0, 1};
        CollComm* collComm = hcclCommPtr->GetCollComm();
        ASSERT_NE(collComm, nullptr);
        ASSERT_EQ(
            HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
                g_fakeWorldTeam, collComm, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2),
            HCCL_SUCCESS);
    }

    // 成功路径默认桩：集中设置 Hcomm/Hccl 接口的成功返回值 + 出参填充。
    void SetupSuccessMocks()
    {
        MOCKER(HcommTeamCreate).stubs().will(invoke(StubSubTeamCreateOk));
        MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
        MOCKER(HcommTeamGetNetLayer).stubs().will(invoke(StubTeamGetNetLayerOne));
        MOCKER(HcommTeamBindChannels).stubs().will(returnValue(0));
        MOCKER(HcommTeamBindRemoteSyncMem).stubs().will(returnValue(0));
        MOCKER(hrtMalloc).stubs().will(invoke(StubHrtMallocFill));
        MOCKER(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&CommMems::CommRegMem).stubs().will(invoke(StubCommRegMemOk));
        MOCKER_CPP(&CommMems::CommUnregMem).stubs().will(invoke(StubCommUnregMem));
        MOCKER(HcclRankGraphGetLinks).stubs().will(invoke(StubRankGraphGetLinks));
        MOCKER(HcclChannelAcquire).stubs().will(invoke(StubChannelAcquireFill));
        MOCKER(HcclChannelGetRemoteMems).stubs().will(invoke(StubChannelGetRemoteMemsSync));
    }

    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    HcclComm comm{nullptr};
};

TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamQuery_When_UbWorldTeamRegistered_Expect_CorrectMapping)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    const uint32_t rankIds[] = {0U, 2U, 4U, 6U};
    ASSERT_EQ(
        HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
            g_fakeWorldTeam, collComm, COMM_PROTOCOL_UB_MEM, 0U, rankIds, 4U),
        HCCL_SUCCESS);

    HcommTeamHandle lsaTeam = nullptr;
    EXPECT_EQ(HcclTeamGetLsaTeam(comm, &lsaTeam), HCCL_SUCCESS);
    EXPECT_EQ(lsaTeam, g_fakeWorldTeam);

    uint32_t rankId = 0U;
    EXPECT_EQ(HcclTeamMemberToRank(comm, lsaTeam, 2U, &rankId), HCCL_SUCCESS);
    EXPECT_EQ(rankId, 4U);

    uint32_t memberId = 0U;
    EXPECT_EQ(HcclTeamRankToMember(comm, lsaTeam, 6U, &memberId), HCCL_SUCCESS);
    EXPECT_EQ(memberId, 3U);
}

TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamGetLsaTeam_When_LsaTeamAbsent_Expect_NotFound)
{
    HcommTeamHandle lsaTeam = g_fakeWorldTeam;
    EXPECT_EQ(HcclTeamGetLsaTeam(comm, &lsaTeam), HCCL_E_NOT_FOUND);
    EXPECT_EQ(lsaTeam, nullptr);
}

TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamQuery_When_ParameterInvalid_Expect_Error)
{
    HcommTeamHandle lsaTeam = g_fakeWorldTeam;
    EXPECT_EQ(HcclTeamGetLsaTeam(nullptr, &lsaTeam), HCCL_E_PTR);
    EXPECT_EQ(lsaTeam, nullptr);
    EXPECT_EQ(HcclTeamGetLsaTeam(comm, nullptr), HCCL_E_PTR);

    uint32_t id = 0U;
    EXPECT_EQ(HcclTeamMemberToRank(nullptr, g_fakeWorldTeam, 0U, &id), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamMemberToRank(comm, nullptr, 0U, &id), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamMemberToRank(comm, g_fakeWorldTeam, 0U, nullptr), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamRankToMember(nullptr, g_fakeWorldTeam, 0U, &id), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamRankToMember(comm, nullptr, 0U, &id), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamRankToMember(comm, g_fakeWorldTeam, 0U, nullptr), HCCL_E_PTR);
}

TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamQuery_When_IdOutsideTeam_Expect_Error)
{
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    const uint32_t rankIds[] = {0U, 2U, 4U, 6U};
    ASSERT_EQ(
        HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
            g_fakeWorldTeam, collComm, COMM_PROTOCOL_UB_MEM, 0U, rankIds, 4U),
        HCCL_SUCCESS);

    uint32_t id = 0U;
    EXPECT_EQ(HcclTeamMemberToRank(comm, g_fakeWorldTeam, 4U, &id), HCCL_E_PARA);
    EXPECT_EQ(HcclTeamRankToMember(comm, g_fakeWorldTeam, 1U, &id), HCCL_E_NOT_FOUND);
}

// =====================================================================
// 1. HcclTeamCreate（ut_ain_001~010）
// =====================================================================

// ut_ain_001 正常路径：预制 worldTeam 命中，一次完成 subTeam 创建 + syncMem + 建链。
// expects 须在 SetupSuccessMocks 前注册（mockcpp FIFO，先注册的先命中）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_Valid_Expect_SuccessAndChannelsBound)
{
    MOCKER(HcommTeamBindChannels).expects(once()).will(returnValue(0));
    MOCKER(HcommTeamBindRemoteSyncMem).expects(once()).will(returnValue(0));
    SetupSuccessMocks();
    RegisterDefaultWorldTeam();

    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0xdeadbeef);

    HcclResult ret = HcclTeamCreate(comm, &desc, &team);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(team, g_fakeSubTeam);
    // HcclTeamMgr 状态：FindWorldTeam 命中预制 worldTeam，syncMemPtr 为 hrtMalloc 桩指针
    EXPECT_EQ(HcclTeamMgr::GetInstance().FindWorldTeam(team), g_fakeWorldTeam);
    EXPECT_EQ(HcclTeamMgr::GetInstance().GetSyncMemPtr(team), g_fakeWsPtr);
    EXPECT_EQ(HcclTeamMgr::GetInstance().GetRankIds(team).size(), 2U);

    // 清理
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_SUCCESS);
}

// ut_ain_002/003 入参空指针：comm/desc/team 出参为空。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_NullptrParams_Expect_ReturnHCCL_E_PTR)
{
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;
    EXPECT_EQ(HcclTeamCreate(nullptr, &desc, &team), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamCreate(comm, nullptr, &team), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamCreate(comm, &desc, nullptr), HCCL_E_PTR);
}

// ut_ain_004 rankNum=0/1（team 不能为空或单 rank）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_SingleOrZeroRank_Expect_ReturnParaError)
{
    HcclTeamCreateDesc desc;
    HcommTeamHandle team = nullptr;
    uint32_t rankIds1[1] = {1};

    (void)HcclTeamCreateDescInit(&desc);
    desc.rankNum = 0;
    desc.rankIds = nullptr;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);

    BuildTeamCreateDesc(desc, rankIds1, 1, 1, 1);
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);
}

// ut_ain_005 rankIds=nullptr（rankNum=2 非零）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_RankIdsNull_Expect_ReturnHCCL_E_PTR)
{
    HcclTeamCreateDesc desc;
    HcommTeamHandle team = nullptr;
    (void)HcclTeamCreateDescInit(&desc);
    desc.rankNum = 2;
    desc.rankIds = nullptr;
    desc.selfRankId = 1;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PTR);
}

// ut_ain_006 requirement 非法：signalCount≠0 / counterCount≠0 / barrierCount=0。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_RequirementInvalid_Expect_ReturnParaError)
{
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    HcommTeamHandle team = nullptr;

    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    desc.requirement.signalCount = 1;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);

    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    desc.requirement.counterCount = 1;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);

    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    desc.requirement.barrierCount = 0;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);
}

// ut_ain_007 channelCnt=0 / protocol=RESERVED / engine≠AIV。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_DescFieldInvalid_Expect_ReturnParaError)
{
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    HcommTeamHandle team = nullptr;

    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    desc.channelCnt = 0;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);

    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    desc.protocol = COMM_PROTOCOL_RESERVED;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);

    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    desc.engine = COMM_ENGINE_AICPU;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);
}

// ut_ain_008 selfRankId 不在 rankIds 中：FindSelfMemberId 失败（发生在 worldTeam 查找之后）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_SelfRankNotInRankIds_Expect_ReturnNotFound)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 5, 1); // selfRankId=5 不在 {0,1}
    HcommTeamHandle team = nullptr;

    // worldTeam 已命中（否则 E_NOT_FOUND 来自 FindWorldTeamByProtoLayer，无法区分），
    // 断言 HcommTeamCreate 未被调（校验发生在 L3 创建之前）。
    MOCKER(HcommTeamCreate).expects(never());
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_NOT_FOUND);
}

// ut_ain_009 (protocol,netLayer) 无预制 worldTeam。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_NoPrebuiltWorldTeam_Expect_ReturnNotFound)
{
    // 不登记任何预制 worldTeam，protocol=UBOE 未预制
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    desc.protocol = COMM_PROTOCOL_UBOE;
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).expects(never());
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_NOT_FOUND);
    EXPECT_EQ(team, nullptr);
}

// ut_ain_010 subTeam rankId 不在 worldTeam rankIds 中：BuildSubTeamWorldMemberIds 失败。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_SubRankIdNotInWorldTeam_Expect_ReturnParaError)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 9}; // rankId=9 不在预制 worldTeam 的 {0,1}
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 0, 1); // selfRankId=0 在 rankIds 中，跳过 FindSelfMemberId
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).expects(never());
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);
}

// =====================================================================
// 2. HcclTeamCreate L3 创建与回滚（ut_ain_011~015）
// =====================================================================

// ut_ain_011 HcommTeamCreate 失败 → 统一 HCCL_E_INTERNAL（82d6b6376 收编，不透传），无回滚。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_HcommCreateFail_Expect_ReturnInternalNoRollback)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).stubs().will(invoke(StubSubTeamCreateFail));
    MOCKER(HcommTeamDestroy).expects(never());
    SetupSuccessMocks();
    HcclResult ret = HcclTeamCreate(comm, &desc, &team);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(team, nullptr);
}

// ut_ain_011 变体：HcommTeamCreate 返回 HCOMM_E_PARA 也收编为 HCCL_E_INTERNAL。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_HcommCreateReturnPara_Expect_ReturnInternal)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_E_PARA)));
    MOCKER(HcommTeamDestroy).expects(never());
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_INTERNAL);
}

// ut_ain_012 HcommTeamCreate 返回 ok 但 *team=nullptr → HCCL_E_INTERNAL。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_StubReturnNullTeam_Expect_ReturnInternal)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).stubs().will(invoke(StubSubTeamCreateNullTeam));
    MOCKER(HcommTeamDestroy).expects(never());
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_INTERNAL);
}

// ut_ain_013 syncMemSize=0 → HCCL_E_PARA；修复后语义：CreateSubTeamWithSyncMem 内部调
// HcommTeamDestroy 回滚已创建的 sub team（不再残留），出参 team 保持 nullptr。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_SyncMemSizeZero_Expect_ReturnParaNoRollback)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).stubs().will(invoke(StubSubTeamCreateZeroWs));
    MOCKER(hrtMalloc).expects(never());
    MOCKER(HcommTeamDestroy).expects(exactly(1)).will(returnValue(0));
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);
    EXPECT_EQ(team, nullptr); // 已回滚，出参清空
}

// ut_ain_014 hrtMalloc 失败（AllocTeamSyncMem）→ HCCL_E_MEMORY；回滚 HcommTeamDestroy；hrtFree 未被调。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_HrtMallocFail_Expect_RollbackAndMemoryError)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).stubs().will(invoke(StubSubTeamCreateOk));
    MOCKER(hrtMalloc).stubs().will(returnValue(HCCL_E_MEMORY));
    MOCKER(HcommTeamDestroy).expects(once()).will(returnValue(0));
    MOCKER(hrtFree).expects(never());
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_MEMORY);
    EXPECT_EQ(team, nullptr);
}

// ut_ain_015 RegisterSubTeam 失败（注入重复 subTeam 句柄）→ 透传 regRet + 完整回滚。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_RegisterSubTeamFail_Expect_RollbackAndReturnRegRet)
{
    RegisterDefaultWorldTeam();
    // 先真实注册 g_fakeSubTeam，使后续 RegisterSubTeam 返回 HCCL_E_PARA
    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    uint32_t preRankIds[2] = {0, 1};
    ASSERT_EQ(
        HcclTeamMgr::GetInstance().RegisterSubTeam(g_fakeWorldTeam, g_fakeSubTeam, nullptr, 0, preRankIds, 2),
        HCCL_SUCCESS);

    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamCreate).stubs().will(invoke(StubSubTeamCreateOk));
    // 回滚：hrtFree(syncMemPtr) 1 次 + HcommTeamDestroy 1 次
    MOCKER(hrtFree).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommTeamDestroy).expects(once()).will(returnValue(0));
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);
    EXPECT_EQ(team, nullptr);
    // 清理预注册条目（TearDown 也会兜底）
    HcclTeamMgr::GetInstance().UnregisterTeam(g_fakeSubTeam);
}

// =====================================================================
// 3. HcclTeamDestroy（ut_ain_016~020、ut_ain_067）
// =====================================================================

// ut_ain_016 正常销毁：CommUnregMem(syncmem tag) + hrtFree + HcommTeamDestroy 均执行。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_TeamRegistered_Expect_SuccessAndWsFreed)
{
    SetupSuccessMocks();
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;
    ASSERT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_SUCCESS);
    GlobalMockObject::verify(); // 清除建链阶段的 mock 计数，避免影响销毁断言

    // 销毁：syncMemHandle 已在 Create 中注册，CommUnregMem 被调 1 次
    g_commUnregMemCallCnt = 0;
    MOCKER_CPP(&CommMems::CommUnregMem).stubs().will(invoke(StubCommUnregMem));
    MOCKER(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_SUCCESS);
    EXPECT_EQ(g_commUnregMemCallCnt, 1U);
    // 条目已 erase
    EXPECT_EQ(HcclTeamMgr::GetInstance().FindCollComm(team), nullptr);
}

// ut_ain_017 team=nullptr。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_TeamNullptr_Expect_ReturnHCCL_E_PTR)
{
    EXPECT_EQ(HcclTeamDestroy(nullptr), HCCL_E_PTR);
}

// ut_ain_018 CommUnregMem 返回 HCCL_E_NOT_FOUND → 容忍（HCCL_WARNING），最终 HCCL_SUCCESS。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_CommUnregMemNotFound_Expect_TolerateSuccess)
{
    SetupSuccessMocks();
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;
    ASSERT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_SUCCESS);
    GlobalMockObject::verify();

    g_commUnregMemRet = HCCL_E_NOT_FOUND;
    g_commUnregMemCallCnt = 0;
    MOCKER_CPP(&CommMems::CommUnregMem).stubs().will(invoke(StubCommUnregMem));
    MOCKER(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_SUCCESS);
    EXPECT_EQ(g_commUnregMemCallCnt, 1U);
}

// ut_ain_019 CommUnregMem 返回其他错误（HCCL_E_INTERNAL）→ 82d6b6376 起 if+WARNING 后继续执行：
// UnregisterTeam/L3 HcommTeamDestroy/hrtFree(syncMemPtr) 不被跳过，最终 HCCL_SUCCESS。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_CommUnregMemFail_Expect_ContinueCleanupAndSuccess)
{
    SetupSuccessMocks();
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;
    ASSERT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_SUCCESS);
    GlobalMockObject::verify();

    g_commUnregMemRet = HCCL_E_INTERNAL;
    g_commUnregMemCallCnt = 0;
    MOCKER_CPP(&CommMems::CommUnregMem).stubs().will(invoke(StubCommUnregMem));
    // 后续清理不被跳过：hrtFree(syncMemPtr) + L3 HcommTeamDestroy 均执行
    MOCKER(hrtFree).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommTeamDestroy).expects(once()).will(returnValue(0));
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_SUCCESS);
    EXPECT_EQ(g_commUnregMemCallCnt, 1U);
    // UnregisterTeam 已执行（条目 erase）
    EXPECT_EQ(HcclTeamMgr::GetInstance().FindCollComm(team), nullptr);
}

// ut_ain_020 重复销毁（幂等）：第二次 SUCCESS。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_DoubleDestroy_Expect_IdempotentSuccess)
{
    SetupSuccessMocks();
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;
    ASSERT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&CommMems::CommUnregMem).stubs().will(invoke(StubCommUnregMem));
    MOCKER(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_SUCCESS);

    // 第二次销毁：条目已 erase，GetTeamSyncMemHandle 空 → 跳过 unreg；UnregisterTeam 空转；
    // L3 HcommTeamDestroy 对未登记句柄桩返回 SUCCESS
    g_commUnregMemCallCnt = 0;
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_SUCCESS);
    EXPECT_EQ(g_commUnregMemCallCnt, 0U); // 第二次不再调 CommUnregMem
}

// ut_ain_067 L3 HcommTeamDestroy 失败收编：无论 L3 返回何错误码均 HCCL_E_INTERNAL；
// CommUnregMem/UnregisterTeam 已先行执行。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_HcommDestroyFail_Expect_ReturnInternal)
{
    SetupSuccessMocks();
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;
    ASSERT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_SUCCESS);
    GlobalMockObject::verify();

    g_commUnregMemRet = HCCL_SUCCESS;
    g_commUnregMemCallCnt = 0;
    MOCKER_CPP(&CommMems::CommUnregMem).stubs().will(invoke(StubCommUnregMem));
    MOCKER(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommTeamDestroy).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_E_INTERNAL)));
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_E_INTERNAL);
    EXPECT_EQ(g_commUnregMemCallCnt, 1U);                              // CommUnregMem 已先行执行
    EXPECT_EQ(HcclTeamMgr::GetInstance().FindCollComm(team), nullptr); // UnregisterTeam 已先行执行
}

// =====================================================================
// 4. 建链段 L3 失败收编 HCCL_E_INTERNAL（ut_ain_066 三变体）
// =====================================================================

// ut_ain_066 ① HcommTeamGetNetLayer 失败（FillChannelDescForPeer 内）→ HCCL_E_INTERNAL。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_GetNetLayerFail_Expect_ReturnInternal)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamGetNetLayer).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_E_INTERNAL)));
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_INTERNAL);
    /* 修复后语义：SetupTeamChannels 失败统一回滚（HcclTeamDestroy 完整逆操作），出参清空 */
    EXPECT_EQ(team, nullptr);
}

// ut_ain_066 ② HcommTeamBindChannels 失败 → HCCL_E_INTERNAL。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_BindChannelsFail_Expect_ReturnInternal)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamBindChannels).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_E_INTERNAL)));
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_INTERNAL);
    HcclTeamMgr::GetInstance().UnregisterTeam(team);
}

// ut_ain_066 ③ HcommTeamBindRemoteSyncMem 失败 → HCCL_E_INTERNAL。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_BindRemoteSyncMemFail_Expect_ReturnInternal)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcommTeamBindRemoteSyncMem).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_E_INTERNAL)));
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_INTERNAL);
    HcclTeamMgr::GetInstance().UnregisterTeam(team);
}

// 对照：HcclRankGraphGetLinks 无 link → HCCL_E_NOT_FOUND（FillChannelDescForPeer 的另一分支）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_NoLink_Expect_ReturnNotFound)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcclRankGraphGetLinks).stubs().will(invoke(StubRankGraphGetLinksEmpty));
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_NOT_FOUND);
    HcclTeamMgr::GetInstance().UnregisterTeam(team);
}

// 对照：HcclChannelAcquire 失败 → 仍透传（不在收编范围）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_ChannelAcquireFail_Expect_ReturnRet)
{
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;

    MOCKER(HcclChannelAcquire).stubs().will(returnValue(HCCL_E_PARA));
    SetupSuccessMocks();
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_E_PARA);
    HcclTeamMgr::GetInstance().UnregisterTeam(team);
}

// =====================================================================
// 5. ut_ain_065：RegisterTeamSyncMem 幂等
// =====================================================================
// 首次 Create 后 syncMemHandle 已登记（SetTeamSyncMemHandle），CollectPendingMemHandles 标记已交换；
// 二次 Create 同一 team（L3 桩返回同句柄）在 RegisterSubTeam 处因重复返回 E_PARA 并完整回滚
// （hrtFree+HcommTeamDestroy），不会重复走建链段。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamCreate_When_SyncMemHandleExists_Expect_CommRegMemSkipped)
{
    SetupSuccessMocks();
    RegisterDefaultWorldTeam();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle team = nullptr;
    ASSERT_EQ(HcclTeamCreate(comm, &desc, &team), HCCL_SUCCESS);
    // 首次建链后 syncMemHandle 已登记（RegisterTeamSyncMem 的幂等依据）
    ASSERT_NE(HcclTeamMgr::GetInstance().GetTeamSyncMemHandle(team), nullptr);
    GlobalMockObject::verify();

    // 二次 Create 同一 team：RegisterSubTeam 判重失败 → 完整回滚，不重复进入建链段
    g_commRegMemCallCnt = 0;
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubSubTeamCreateOk));
    MOCKER(hrtMalloc).stubs().will(invoke(StubHrtMallocFill));
    MOCKER_CPP(&CommMems::CommRegMem).stubs().will(invoke(StubCommRegMemCountFail));
    MOCKER_CPP(&CommMems::CommUnregMem).stubs().will(invoke(StubCommUnregMem));
    MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
    MOCKER(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    HcommTeamHandle team2 = nullptr;
    EXPECT_EQ(HcclTeamCreate(comm, &desc, &team2), HCCL_E_PARA);
    EXPECT_EQ(team2, nullptr);

    // 清理首次创建的 team（CommUnregMem 幂等计数）
    EXPECT_EQ(HcclTeamDestroy(team), HCCL_SUCCESS);
    EXPECT_EQ(g_commRegMemCallCnt, 0U); // 二次 Create 未走到 RegisterTeamSyncMem（建链段未重复执行）
}

TEST_F(TestCollCommTeamCAdpt, Ut_HcclCommMemReg_When_TagPrefixes_Expect_SymAndSyncmemRejectedUsermemAllowed)
{
    // 本用例为 PR 82d6b6376 的 HcclCommMemReg memTag 前缀校验回归：
    // __hccl_sym_win__/__hccl_team_syncmem__ 为保留前缀（HCCL_E_PARA），__hccl_team_usermem__ 放行。
    // HcclCommMemReg 完整参数校验用例见 coll_comms/rank/comm_mems/test_hcclCommMemReg_api.cc。
    // __hccl_sym_win__ 前缀：保留，HCCL_E_PARA
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = reinterpret_cast<void*>(0x31000);
        mem.size = 1024;
        HcclMemHandle handle = nullptr;
        EXPECT_EQ(HcclCommMemReg(comm, "__hccl_sym_win__ut", &mem, &handle), HCCL_E_PARA);
    }
    // __hccl_team_syncmem__ 前缀：保留，HCCL_E_PARA
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = reinterpret_cast<void*>(0x32000);
        mem.size = 1024;
        HcclMemHandle handle = nullptr;
        EXPECT_EQ(HcclCommMemReg(comm, "__hccl_team_syncmem__ut", &mem, &handle), HCCL_E_PARA);
    }
    // __hccl_team_usermem__ 前缀：本 PR 起允许（不再被前缀校验拦截），走后续注册流程
    {
        MOCKER_CPP(&CommMems::CommRegMem).stubs().will(invoke(StubCommRegMemOk));
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = reinterpret_cast<void*>(0x33000);
        mem.size = 1024;
        HcclMemHandle handle = nullptr;
        EXPECT_EQ(HcclCommMemReg(comm, "__hccl_team_usermem__ut", &mem, &handle), HCCL_SUCCESS);
    }
}
