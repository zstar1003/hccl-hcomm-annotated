/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <shared_mutex>

#include "ut_hcomm_team_stub.h"

using namespace hcomm;
using namespace hcomm_ut;

// 构造 memberNum 个 device 侧 CommMem（malloc 内存，size 可指定），调用方负责 free 各 addr。
static std::vector<CommMem> MakeDeviceMems(uint32_t memberNum, size_t memSize = 1024)
{
    std::vector<CommMem> mems(memberNum);
    for (uint32_t i = 0; i < memberNum; i++) {
        mems[i].addr = malloc(memSize);
        mems[i].size = memSize;
        mems[i].type = COMM_MEM_TYPE_DEVICE;
    }
    return mems;
}

// 构造 BindSyncMem 描述（memberNum 个 device mem），析构自动 free——用例内直接用。
struct SyncMemBindBundle {
    HcommTeamBindSyncMemDesc desc{};
    std::vector<CommMem> mems;

    explicit SyncMemBindBundle(uint32_t memberNum) : mems(MakeDeviceMems(memberNum))
    {
        desc.remoteMems = mems.data();
        desc.remoteMemNum = memberNum;
    }
    ~SyncMemBindBundle()
    {
        for (auto& mem : mems) {
            free(mem.addr);
        }
    }
    SyncMemBindBundle(const SyncMemBindBundle&) = delete;
    SyncMemBindBundle& operator=(const SyncMemBindBundle&) = delete;
};

// 构造 BindChannels 描述：memberNum 个成员、每成员 1 条 channel，句柄基址 baseAddr 起每条递增 0x100。
// 返回 desc 与持有的 channels/chPtrs（chPtrs 生命周期需覆盖 desc 使用）。
struct ChannelsDescBundle {
    HcommTeamBindChannelsDesc desc{};
    std::vector<uint32_t> chNums;
    std::vector<std::vector<uint64_t>> channels;
    std::vector<uint64_t*> chPtrs;
};

static ChannelsDescBundle MakeChannelsDesc(uint32_t memberNum, uint64_t baseAddr)
{
    ChannelsDescBundle bundle;
    bundle.chNums.assign(memberNum, 1);
    bundle.channels.assign(memberNum, {});
    for (uint32_t i = 0; i < memberNum; i++) {
        bundle.channels[i] = {baseAddr + i * 0x100};
    }
    for (auto& ch : bundle.channels) {
        bundle.chPtrs.push_back(ch.data());
    }
    bundle.desc.memberNum = memberNum;
    bundle.desc.channelNumPerMember = bundle.chNums.data();
    bundle.desc.channelsByMemberId = bundle.chPtrs.data();
    return bundle;
}

// 构造标准 HcommTeamCreateDesc（memberNum=4/netLayer=1/barrierCount=1），worldMemberIds 指向 ids。
struct TeamCreateDescBundle {
    HcommTeamCreateDesc desc{};
    std::vector<uint32_t> ids;
};

static TeamCreateDescBundle MakeTeamCreateDesc(uint32_t memberNum)
{
    TeamCreateDescBundle bundle;
    bundle.ids.resize(memberNum);
    for (uint32_t i = 0; i < memberNum; i++) {
        bundle.ids[i] = i;
    }
    bundle.desc.memberNum = memberNum;
    bundle.desc.selfMemberId = 0;
    bundle.desc.worldMemberIds = bundle.ids.data();
    bundle.desc.netLayer = 1;
    bundle.desc.requirement = {0, 0, 1, {}};
    return bundle;
}

class TestHcommTeam : public hcomm_ut::HcommTeamUtBase {
public:
    HcommResult CreateWorldTeam(uint32_t memberNum, uint32_t selfMemberId, HcommTeamHandle& team, uint64_t& syncMemSize)
    {
        return CreateWorldTeam(memberNum, selfMemberId, 1, team, syncMemSize);
    }

    HcommResult CreateWorldTeam(
        uint32_t memberNum, uint32_t selfMemberId, uint32_t netLayer, HcommTeamHandle& team, uint64_t& syncMemSize)
    {
        std::vector<uint32_t> ids(memberNum);
        for (uint32_t i = 0; i < memberNum; i++) {
            ids[i] = i;
        }
        HcommTeamCreateDesc desc{};
        desc.memberNum = memberNum;
        desc.selfMemberId = selfMemberId;
        desc.worldMemberIds = ids.data();
        desc.netLayer = netLayer;
        desc.requirement = {0, 0, 1, {}};
        return HcommTeamCreate(nullptr, &desc, &team, &syncMemSize);
    }

    void BindValidChannels(HcommTeamHandle team, uint32_t memberNum)
    {
        std::vector<uint32_t> chNums(memberNum, 1);
        std::vector<std::vector<uint64_t>> channels(memberNum, std::vector<uint64_t>{0x100});
        std::vector<uint64_t*> chPtrs;
        for (auto& ch : channels) {
            chPtrs.push_back(ch.data());
        }
        HcommTeamBindChannelsDesc desc{};
        desc.memberNum = memberNum;
        desc.channelNumPerMember = chNums.data();
        desc.channelsByMemberId = chPtrs.data();
        HcommTeamBindChannels(team, &desc);
    }

    void BindValidSyncMem(HcommTeamHandle team, uint32_t memberNum)
    {
        std::vector<CommMem> mems = MakeDeviceMems(memberNum);
        HcommTeamBindSyncMemDesc desc{};
        desc.remoteMems = mems.data();
        desc.remoteMemNum = memberNum;
        HcommTeamBindRemoteSyncMem(team, &desc);
        for (uint32_t i = 0; i < memberNum; i++) {
            free(mems[i].addr);
        }
    }

    // 新 API：按 remoteRank 回填远端内存（slots 为各层槽位，这里用 0..memberNum-1 单层简化）
    void BindValidWindow(HcclCommSymWindow handle, uint32_t memberNum)
    {
        std::vector<uint32_t> sizes = {memberNum, 0, 0};
        std::vector<uint32_t> slots = {0};
        for (uint32_t rank = 0; rank < memberNum; rank++) {
            slots[0] = rank;
            DeviceMem mem;
            ASSERT_EQ(
                HcommTeamUpdateWindowRemoteMemByRank(
                    handle, sizes.data(), static_cast<uint32_t>(sizes.size()), slots.data(), 1, &mem),
                HCOMM_SUCCESS);
        }
    }
};

// ==================== ut_team_001 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_8MembersValid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    HcommResult ret = CreateWorldTeam(8, 3, team, syncMemSize);
    EXPECT_EQ(ret, HCOMM_SUCCESS);
    EXPECT_NE(team, nullptr);
    EXPECT_EQ(syncMemSize, static_cast<uint64_t>((0 + 0 + 1) * sizeof(uint64_t) * 8));
}

// ==================== ut_team_002 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_MemberNum1_Expect_ReturnParaError)
{
    std::vector<uint32_t> ids = {0};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 1;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.netLayer = 0;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_E_PARA);
    EXPECT_EQ(team, nullptr);
}

// ==================== ut_team_003 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_DescNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, nullptr, &team, &syncMemSize), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_TeamOutNull_Expect_ReturnPtrError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 2;
    desc.selfMemberId = 0;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, nullptr, &syncMemSize), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SyncMemSizeOutNull_Expect_ReturnPtrError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 2;
    desc.selfMemberId = 0;
    HcommTeamHandle team = nullptr;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, nullptr), HCOMM_E_PTR);
}

// ==================== ut_team_004 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_MemberNumZero_Expect_ReturnParaError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 0;
    desc.selfMemberId = 0;
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_E_PARA);
}

// ==================== ut_team_005 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SelfMemberIdExceeds_Expect_ReturnParaError)
{
    std::vector<uint32_t> ids = {0, 1, 2, 3};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 4;
    desc.worldMemberIds = ids.data();
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_E_PARA);
}

// ==================== ut_team_006 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamWorldMemberIdsNull_Expect_ReturnPtrError)
{
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 0;
    desc.worldMemberIds = nullptr;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    HcommTeamHandle fakeWorldTeam = reinterpret_cast<HcommTeamHandle>(0x10000);
    EXPECT_EQ(HcommTeamCreate(fakeWorldTeam, &desc, &team, &syncMemSize), HCOMM_E_PTR);
}

// ==================== ut_team_007 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamValid_Expect_ReturnSuccess)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(8, 3, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> subIds = {2, 5, 7};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 3;
    desc.selfMemberId = 1;
    desc.worldMemberIds = subIds.data();
    desc.netLayer = 0;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle subTeam = nullptr;
    uint64_t subWsSize = 0;
    EXPECT_EQ(HcommTeamCreate(worldTeam, &desc, &subTeam, &subWsSize), HCOMM_SUCCESS);
    EXPECT_NE(subTeam, nullptr);
}

// ==================== ut_team_008 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamInvalidWorldTeam_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fakeWorld = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    std::vector<uint32_t> ids = {0, 1, 2};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 3;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_EQ(HcommTeamCreate(fakeWorld, &desc, &team, &syncMemSize), HCOMM_E_NOT_FOUND);
}

// ==================== ut_team_009 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamMemberNumExceeds_Expect_ReturnParaError)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> ids(8, 0);
    HcommTeamCreateDesc desc{};
    desc.memberNum = 8;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    HcommTeamHandle team = nullptr;
    uint64_t wsSize2 = 0;
    EXPECT_EQ(HcommTeamCreate(worldTeam, &desc, &team, &wsSize2), HCOMM_E_PARA);
}

// ==================== ut_team_010 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SubTeamWorldMemberIdsOutOfBounds_Expect_ReturnParaError)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> ids = {1, 3, 5};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 3;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    HcommTeamHandle team = nullptr;
    uint64_t wsSize2 = 0;
    EXPECT_EQ(HcommTeamCreate(worldTeam, &desc, &team, &wsSize2), HCOMM_E_PARA);
}

// ==================== ut_team_011 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.find(team), mgr.teams_.end());
}

// ==================== ut_team_012 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_Null_Expect_ReturnPtrError)
{
    EXPECT_EQ(HcommTeamDestroy(nullptr), HCOMM_E_PTR);
}

// 重复销毁（handle 不存在）幂等返回 SUCCESS，仅打 WARNING
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_InvalidHandle_Expect_ReturnSuccess)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    EXPECT_EQ(HcommTeamDestroy(fake), HCOMM_SUCCESS);
}

// HcommTeamMgr 的 WindowRegister/WindowDeregister/UpdateWindowRemoteMemByRank 用例
// 已按源文件对应迁至 ut_hcomm_team_mgr.cc（含各 C 入口的 nullptr 判空分支）；
// ut_team_035 场景（window 注销先于 team 销毁）随之迁移。

// ==================== ut_team_021 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_FirstBindValid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> chNums = {2, 2, 2, 2};
    std::vector<std::vector<uint64_t>> channels = {{0x100, 0x200}, {0x300, 0x400}, {0x500, 0x600}, {0x700, 0x800}};
    std::vector<uint64_t*> chPtrs;
    for (auto& ch : channels) {
        chPtrs.push_back(ch.data());
    }
    HcommTeamBindChannelsDesc desc{};
    desc.memberNum = 4;
    desc.channelNumPerMember = chNums.data();
    desc.channelsByMemberId = chPtrs.data();
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto* entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->channelsList.size(), 4u);
    EXPECT_EQ(entry->channelsList[0].size(), 2u);
    // devChannels 是 device 侧连续 channel 数组基地址；channelCntAccumulatePerMember 是 device 侧每成员 channel
    // 前缀和数组。
    EXPECT_NE(entry->devChannels, nullptr);
    EXPECT_NE(entry->hostTeam.channelCntAccumulatePerMember, nullptr);
}

// ==================== ut_team_022 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_SecondAppend_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    ChannelsDescBundle bundle1 = MakeChannelsDesc(4, 0x100);
    ASSERT_EQ(HcommTeamBindChannels(team, &bundle1.desc), HCOMM_SUCCESS);

    ChannelsDescBundle bundle2 = MakeChannelsDesc(4, 0x500);
    EXPECT_EQ(HcommTeamBindChannels(team, &bundle2.desc), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto* entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->channelsList[0].size(), 2u);
    EXPECT_EQ(entry->channelsList[0][0], 0x100u);
    EXPECT_EQ(entry->channelsList[0][1], 0x500u);
}

// ==================== ut_team_023 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_TeamNull_Expect_ReturnPtrError)
{
    HcommTeamBindChannelsDesc desc{};
    EXPECT_EQ(HcommTeamBindChannels(nullptr, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_DescNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamBindChannels(team, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_ChannelNumPerMemberNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommTeamBindChannelsDesc desc{};
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_ChannelsByMemberIdNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    HcommTeamBindChannelsDesc desc{};
    std::vector<uint32_t> chNums = {1, 1};
    desc.channelNumPerMember = chNums.data();
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_MemberNumMismatch_Expect_ReturnParaError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommTeamBindChannelsDesc desc{};
    desc.memberNum = 8;
    std::vector<uint32_t> chNums(8, 1);
    std::vector<std::vector<uint64_t>> channels(8, std::vector<uint64_t>{0x100});
    std::vector<uint64_t*> chPtrs;
    for (auto& ch : channels) {
        chPtrs.push_back(ch.data());
    }
    desc.channelNumPerMember = chNums.data();
    desc.channelsByMemberId = chPtrs.data();
    EXPECT_EQ(HcommTeamBindChannels(team, &desc), HCOMM_E_PARA);
}

// ==================== ut_team_025 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    SyncMemBindBundle bundle(4);
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &bundle.desc), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto* entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_NE(entry->devRemoteMems, nullptr);
    EXPECT_NE(entry->hostRemoteMems, nullptr);
    EXPECT_EQ(entry->hostTeam.syncMem.remoteMemsNum, 4u);
}

// ==================== ut_team_026 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_ReplaceOld_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    SyncMemBindBundle bundle1(4);
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &bundle1.desc), HCOMM_SUCCESS);

    SyncMemBindBundle bundle2(4);
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &bundle2.desc), HCOMM_SUCCESS);
}

// ==================== ut_team_027 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_TeamNull_Expect_ReturnPtrError)
{
    HcommTeamBindSyncMemDesc desc{};
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(nullptr, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_DescNull_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_RemoteMemsNull_Expect_ReturnPtrError)
{
    // 源码：remoteMems 数组下标为 memberId、长度=memberNum（恒>0），故 remoteMems 必须非空，为 nullptr 返回
    // HCOMM_E_PTR。
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = nullptr;
    desc.remoteMemNum = 4;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_RemoteMemNumMismatch_Expect_ReturnParaError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    std::vector<CommMem> mems(8);
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 8;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_E_PARA);
}

// ==================== ut_team_030 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_Valid_Expect_ReturnSuccess)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, 1u, team, syncMemSize), HCOMM_SUCCESS);

    uint32_t netLayer = 0;
    EXPECT_EQ(HcommTeamGetNetLayer(team, &netLayer), HCOMM_SUCCESS);
    EXPECT_EQ(netLayer, 1u);
}

// ==================== ut_team_031 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_Null_Expect_ReturnPtrError)
{
    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamGetNetLayer(team, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_TeamNull_Expect_ReturnPtrError)
{
    uint32_t netLayer = 0;
    EXPECT_EQ(HcommTeamGetNetLayer(nullptr, &netLayer), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamGetNetLayer_When_InvalidHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    uint32_t netLayer = 0;
    EXPECT_EQ(HcommTeamGetNetLayer(fake, &netLayer), HCOMM_E_NOT_FOUND);
}

// ==================== ut_team_032 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_HrtMallocFailAfterWorldTeamIds_Expect_NoLeak)
{
    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 1;

    std::vector<uint32_t> ids = {0, 1, 2, 3};
    HcommTeamCreateDesc desc{};
    desc.memberNum = 4;
    desc.selfMemberId = 0;
    desc.worldMemberIds = ids.data();
    desc.netLayer = 1;
    desc.requirement = {0, 0, 1, {}};
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    HcommResult ret = HcommTeamCreate(nullptr, &desc, &team, &syncMemSize);
    EXPECT_NE(ret, HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.size(), 0u);
}

// ==================== ut_team_033 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_AppendFail_Expect_Rollback)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    ChannelsDescBundle bundle1 = MakeChannelsDesc(4, 0x100);
    ASSERT_EQ(HcommTeamBindChannels(team, &bundle1.desc), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::vector<std::vector<uint64_t>> oldChannels;
    {
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto* entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        oldChannels = entry->channelsList;
    }

    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    ChannelsDescBundle bundle2 = MakeChannelsDesc(4, 0x500);
    HcommResult ret = HcommTeamBindChannels(team, &bundle2.desc);
    EXPECT_NE(ret, HCOMM_SUCCESS);

    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto* entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->channelsList.size(), oldChannels.size());
    for (size_t i = 0; i < oldChannels.size(); i++) {
        EXPECT_EQ(entry->channelsList[i], oldChannels[i]);
    }
}

// ==================== ut_team_034 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_BoundResources_Expect_AllFreed)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    BindValidChannels(team, 4);
    BindValidSyncMem(team, 4);

    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);
    BindValidWindow(handle, 4);

    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.find(team), mgr.teams_.end());
    free(devSymWin);
}

// ==================== ut_team_036 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_WindowExists_Expect_WindowDeregisterTolerated)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    // team 销毁不再连带 window（window 归通信域管理）；先销毁 team 后注销 window 仍成功
    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamWindowDeregister(handle), HCOMM_SUCCESS);
    free(devSymWin);
}

// ==================== ut_team_037 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_MultipleTeamsWindowsMixed_Expect_AllSucceed)
{
    HcommTeamHandle team1 = nullptr, team2 = nullptr;
    uint64_t wsSize1 = 0, wsSize2 = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team1, wsSize1), HCOMM_SUCCESS);
    ASSERT_EQ(CreateWorldTeam(8, 0, team2, wsSize2), HCOMM_SUCCESS);

    void* devWin1 = AllocFakeWindow();
    HcclCommSymWindow win1 = nullptr;
    void* devWin2 = AllocFakeWindow();
    HcclCommSymWindow win2 = nullptr;
    void* devWin3 = AllocFakeWindow();
    HcclCommSymWindow win3 = nullptr;
    void* devWin4 = AllocFakeWindow();
    HcclCommSymWindow win4 = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devWin1, &win1), HCOMM_SUCCESS);
    ASSERT_EQ(HcommTeamWindowRegister(devWin2, &win2), HCOMM_SUCCESS);
    ASSERT_EQ(HcommTeamWindowRegister(devWin3, &win3), HCOMM_SUCCESS);
    ASSERT_EQ(HcommTeamWindowRegister(devWin4, &win4), HCOMM_SUCCESS);

    EXPECT_EQ(HcommTeamWindowDeregister(win1), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(team1), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamWindowDeregister(win3), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(team2), HCOMM_SUCCESS);
    free(devWin1);
    free(devWin2);
    free(devWin3);
    free(devWin4);
}

// ==================== ut_team_038 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_StructLayout_Expect_FieldOrderCorrect)
{
    EXPECT_LT(offsetof(HcommTeam, header), offsetof(HcommTeam, engine));
    EXPECT_LT(offsetof(HcommTeam, engine), offsetof(HcommTeam, memberNum));
    EXPECT_LT(offsetof(HcommTeam, memberNum), offsetof(HcommTeam, selfMemberId));
    EXPECT_LT(offsetof(HcommTeam, selfMemberId), offsetof(HcommTeam, channelsBaseAddr));
    EXPECT_LT(offsetof(HcommTeam, channelsBaseAddr), offsetof(HcommTeam, channelCntAccumulatePerMember));
    EXPECT_LT(offsetof(HcommTeam, channelCntAccumulatePerMember), offsetof(HcommTeam, netLayer));
    EXPECT_LT(offsetof(HcommTeam, netLayer), offsetof(HcommTeam, worldTeamIds));
    EXPECT_LT(offsetof(HcommTeam, worldTeamIds), offsetof(HcommTeam, syncMem));
    EXPECT_LT(offsetof(HcommTeam, syncMem), offsetof(HcommTeam, reserved));
}

// ==================== ut_team_039 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_HcommWindowStructLayout_Expect_FieldOrderCorrect)
{
    EXPECT_LT(offsetof(HcommWindow, header), offsetof(HcommWindow, netWin.baseRemoteMemAddr));
    EXPECT_LT(offsetof(HcommWindow, netWin.baseRemoteMemAddr), offsetof(HcommWindow, netWin.windowSize));
    EXPECT_LT(offsetof(HcommWindow, netWin.windowSize), offsetof(HcommWindow, netWin.worldTeamAccumulateId));
    EXPECT_LT(offsetof(HcommWindow, netWin.worldTeamAccumulateId), offsetof(HcommWindow, netWin.netLayerNum));
    EXPECT_LT(offsetof(HcommWindow, netWin.netLayerNum), offsetof(HcommWindow, netWin.reserved));
    EXPECT_LT(offsetof(HcommWindow, netWin.reserved), offsetof(HcommWindow, lsaWin.baseVa));
    EXPECT_LT(offsetof(HcommWindow, lsaWin.baseVa), offsetof(HcommWindow, lsaWin.stride));
    EXPECT_LT(offsetof(HcommWindow, lsaWin.stride), offsetof(HcommWindow, lsaWin.userSize));
    EXPECT_LT(offsetof(HcommWindow, lsaWin.userSize), offsetof(HcommWindow, lsaWin.reserved));
    EXPECT_LT(offsetof(HcommWindow, lsaWin.reserved), offsetof(HcommWindow, legacySymWindow));
    EXPECT_LT(offsetof(HcommWindow, legacySymWindow), offsetof(HcommWindow, reserved));
}

// ==================== ut_team_040 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_ConcurrentCreate_Expect_NoDataRace)
{
    const int threadNum = 4;
    std::vector<HcommTeamHandle> teams(threadNum, nullptr);
    std::vector<uint64_t> wsSizes(threadNum, 0);
    std::vector<HcommResult> results(threadNum, HCOMM_E_INTERNAL);

    std::vector<std::thread> threads;
    for (int i = 0; i < threadNum; i++) {
        threads.emplace_back([this, i, &teams, &wsSizes, &results]() {
            results[i] = CreateWorldTeam(4, 0, teams[i], wsSizes[i]);
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < threadNum; i++) {
        EXPECT_EQ(results[i], HCOMM_SUCCESS);
        EXPECT_NE(teams[i], nullptr);
    }
}

// ==================== ut_team_041 ====================
TEST_F(TestHcommTeam, Ut_HcommTeam_When_ConcurrentCreateDestroy_Expect_NoCrash)
{
    const int threadNum = 4;
    std::vector<HcommResult> results(threadNum, HCOMM_E_INTERNAL);

    std::vector<std::thread> threads;
    for (int i = 0; i < threadNum; i++) {
        threads.emplace_back([this, i, &results]() {
            HcommTeamHandle team = nullptr;
            uint64_t syncMemSize = 0;
            HcommResult ret = CreateWorldTeam(4, 0, team, syncMemSize);
            if (ret == HCOMM_SUCCESS && team != nullptr) {
                ret = HcommTeamDestroy(team);
            }
            results[i] = ret;
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < threadNum; i++) {
        EXPECT_EQ(results[i], HCOMM_SUCCESS);
    }
}

// UpdateWindowRemoteMemByRank 的 HcommTeamMgr 层用例（含首次分配/增量回填/多 window 独立性/
// 失败注入）已按源文件对应迁至 ut_hcomm_team_mgr.cc。

// ==================== ut_team_043 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_AfterBind_Expect_DeviceSyncMemValid)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    SyncMemBindBundle bundle(4);
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &bundle.desc), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    auto* entry = mgr.FindTeamByHandleLocked(team);
    ASSERT_NE(entry, nullptr);
    EXPECT_NE(entry->hostTeam.syncMem.remoteMems, nullptr);
    EXPECT_EQ(entry->hostTeam.syncMem.remoteMemsNum, 4u);
    EXPECT_NE(entry->devRemoteMems, nullptr);
}

// ==================== ut_team_044 ====================
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_SubTeamOnly_Expect_WorldTeamAccessible)
{
    HcommTeamHandle worldTeam = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(8, 3, worldTeam, syncMemSize), HCOMM_SUCCESS);

    std::vector<uint32_t> subIds = {2, 5, 7};
    HcommTeamCreateDesc subDesc{};
    subDesc.memberNum = 3;
    subDesc.selfMemberId = 1;
    subDesc.worldMemberIds = subIds.data();
    subDesc.netLayer = 0;
    subDesc.requirement = {0, 0, 1, {}};
    HcommTeamHandle subTeam = nullptr;
    uint64_t subWsSize = 0;
    ASSERT_EQ(HcommTeamCreate(worldTeam, &subDesc, &subTeam, &subWsSize), HCOMM_SUCCESS);

    EXPECT_EQ(HcommTeamDestroy(subTeam), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamDestroy(worldTeam), HCOMM_SUCCESS);
}

// ==================== Additional coverage tests ====================
// ====================

// Create 失败注入公共断言：标准 desc 调 HcommTeamCreate 应失败且单例无残留。
static void ExpectCreateFailAndNoTeamLeft()
{
    TeamCreateDescBundle bundle = MakeTeamCreateDesc(4);
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    EXPECT_NE(HcommTeamCreate(nullptr, &bundle.desc, &team, &syncMemSize), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
    EXPECT_EQ(mgr.teams_.size(), 0u);
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_HrtMallocFailForDevWorldTeamIds_Expect_ReturnError)
{
    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;
    ExpectCreateFailAndNoTeamLeft();
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_HrtMemSyncCopyFailInWorldTeamIds_Expect_ReturnError)
{
    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 0;
    ExpectCreateFailAndNoTeamLeft();
}

TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_SyncTeamToDeviceFail_Expect_ReturnError)
{
    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 1;
    ExpectCreateFailAndNoTeamLeft();
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_HrtMallocFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    std::vector<CommMem> mems(4);
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 4;
    EXPECT_NE(HcommTeamBindRemoteSyncMem(team, &desc), HCOMM_SUCCESS);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_SyncTeamToDeviceFail_Expect_ReturnError)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    SyncMemBindBundle bundle(4);
    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 1;
    EXPECT_NE(HcommTeamBindRemoteSyncMem(team, &bundle.desc), HCOMM_SUCCESS);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindChannels_When_InvalidTeamHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    HcommTeamBindChannelsDesc desc{};
    std::vector<uint32_t> chNums = {1, 1};
    desc.memberNum = 2;
    desc.channelNumPerMember = chNums.data();
    std::vector<uint64_t> ch0 = {0x100};
    std::vector<uint64_t> ch1 = {0x200};
    std::vector<uint64_t*> chPtrs = {ch0.data(), ch1.data()};
    desc.channelsByMemberId = chPtrs.data();
    EXPECT_EQ(HcommTeamBindChannels(fake, &desc), HCOMM_E_NOT_FOUND);
}

TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_InvalidTeamHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    std::vector<CommMem> mems(2);
    HcommTeamBindSyncMemDesc desc{};
    desc.remoteMems = mems.data();
    desc.remoteMemNum = 2;
    EXPECT_EQ(HcommTeamBindRemoteSyncMem(fake, &desc), HCOMM_E_NOT_FOUND);
}

TEST_F(TestHcommTeam, Ut_HcommTeam_When_HcommTeamSyncMemStructLayout_Expect_FieldOrderCorrect)
{
    EXPECT_LT(offsetof(HcommTeamSyncMem, remoteMems), offsetof(HcommTeamSyncMem, remoteMemsNum));
    EXPECT_LT(offsetof(HcommTeamSyncMem, remoteMemsNum), offsetof(HcommTeamSyncMem, syncMemReq));
    EXPECT_LT(offsetof(HcommTeamSyncMem, syncMemReq), offsetof(HcommTeamSyncMem, syncMemSize));
    EXPECT_LT(offsetof(HcommTeamSyncMem, syncMemSize), offsetof(HcommTeamSyncMem, reserved));
}

// ==================== ut_team_046 ====================
// 验证重复 BindSyncMem 时旧 shadowMem 被释放，不会因覆盖而泄漏
TEST_F(TestHcommTeam, Ut_HcommTeamBindRemoteSyncMem_When_Rebind_Expect_OldShadowMemFreed)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    SyncMemBindBundle bundle(4);
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &bundle.desc), HCOMM_SUCCESS);
    auto& mgr = HcommTeamMgr::GetInstance();
    void* firstShadow = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto* entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        firstShadow = entry->hostTeam.syncMem.shadowMem.addr;
        ASSERT_NE(firstShadow, nullptr);
    }
    uint32_t freeBeforeRebind = g_hrtFreeCallCount;

    // 重复 bind：旧 shadowMem 必须先被释放，再申请新的
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &bundle.desc), HCOMM_SUCCESS);
    void* secondShadow = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto* entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        secondShadow = entry->hostTeam.syncMem.shadowMem.addr;
        ASSERT_NE(secondShadow, nullptr);
    }
    // 重复 bind 至少触发一次额外 free（释放旧 shadowMem）
    EXPECT_GT(g_hrtFreeCallCount, freeBeforeRebind);
}

// ==================== ut_team_047 ====================
// 验证 BindSyncMem 后 HcommTeamDestroy 会释放 shadowMem
TEST_F(TestHcommTeam, Ut_HcommTeamDestroy_When_ShadowMemBound_Expect_ShadowMemFreed)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    ASSERT_EQ(CreateWorldTeam(4, 0, team, syncMemSize), HCOMM_SUCCESS);

    SyncMemBindBundle bundle(4);
    ASSERT_EQ(HcommTeamBindRemoteSyncMem(team, &bundle.desc), HCOMM_SUCCESS);
    // 确认 shadowMem 已分配
    {
        auto& mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto* entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        ASSERT_NE(entry->hostTeam.syncMem.shadowMem.addr, nullptr);
    }
    uint32_t freeBeforeDestroy = g_hrtFreeCallCount;

    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);
    // destroy 内 FreeTeamResources 会释放 shadowMem，free 计数应增加
    EXPECT_GT(g_hrtFreeCallCount, freeBeforeDestroy);
}

// ==================== ut_ain_044~046 ====================
// L3 HcommTeamCreate 预制路径（worldTeam=nullptr）+ GetEngine
TEST_F(TestHcommTeam, Ut_HcommTeamCreate_When_PrebuiltPath_Expect_WorldTeamIdsSequential)
{
    HcommTeamHandle team = nullptr;
    uint64_t syncMemSize = 0;
    HcommTeamCreateDesc desc{};
    desc.memberNum = 2;
    desc.selfMemberId = 1;
    desc.worldMemberIds = nullptr;
    desc.netLayer = 1;
    desc.protocol = COMM_PROTOCOL_UB_CTP;
    desc.engine = COMM_ENGINE_AIV;
    desc.requirement = {0, 0, 1, {}};
    ASSERT_EQ(HcommTeamCreate(nullptr, &desc, &team, &syncMemSize), HCOMM_SUCCESS);

    // syncMemSize = (0+0+1)*8*2 = 16
    EXPECT_EQ(syncMemSize, 16U);
    // worldTeamIds 为 [0,1] 连续序列（AllocAndCopyWorldTeamIds src=nullptr 分支）
    {
        auto& mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        auto* entry = mgr.FindTeamByHandleLocked(team);
        ASSERT_NE(entry, nullptr);
        ASSERT_NE(entry->hostWorldTeamIds, nullptr);
        EXPECT_EQ(entry->hostWorldTeamIds[0], 0U);
        EXPECT_EQ(entry->hostWorldTeamIds[1], 1U);
        EXPECT_EQ(entry->hostTeam.engine, COMM_ENGINE_AIV);
        EXPECT_FALSE(entry->isSubTeam);
    }

    // ut_ain_045：GetEngine 正常
    CommEngine engine = COMM_ENGINE_RESERVED;
    EXPECT_EQ(HcommTeamGetEngine(team, &engine), HCOMM_SUCCESS);
    EXPECT_EQ(engine, COMM_ENGINE_AIV);

    EXPECT_EQ(HcommTeamDestroy(team), HCOMM_SUCCESS);
}

TEST_F(TestHcommTeam, Ut_HcommTeamGetEngine_When_NullParams_Expect_ReturnPtrError)
{
    CommEngine engine = COMM_ENGINE_RESERVED;
    EXPECT_EQ(HcommTeamGetEngine(nullptr, &engine), HCOMM_E_PTR);

    HcommTeamHandle team = reinterpret_cast<HcommTeamHandle>(0x1000);
    EXPECT_EQ(HcommTeamGetEngine(team, nullptr), HCOMM_E_PTR);
}

TEST_F(TestHcommTeam, Ut_HcommTeamGetEngine_When_InvalidHandle_Expect_ReturnNotFoundError)
{
    HcommTeamHandle fake = reinterpret_cast<HcommTeamHandle>(0xDEADBEEF);
    CommEngine engine = COMM_ENGINE_RESERVED;
    EXPECT_EQ(HcommTeamGetEngine(fake, &engine), HCOMM_E_NOT_FOUND);
}
