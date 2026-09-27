/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "coll_comm_mgr.h"
extern thread_local s32 g_hcclDeviceId;

// 内部函数（未在头文件中声明），为 UT 直接验证做前向声明
HcclResult ValidateAndInitRootInfoScalable(
    uint32_t nRanks, uint32_t nRoot, uint32_t rank, uint32_t nExtRoot, const HcclCommConfig* config, HcclComm* comm,
    const HcclRootInfo* rootInfoList);
HcclResult
ParseScalableRootHandles(const HcclRootInfo* rootInfoList, u32 nRoot, std::vector<HcclScalableRootHandle>& rootHandles);
HcclResult BuildScalableMeshInfo(
    u32 nRoot, u32 rootIndex, u32 groupSize, const std::vector<HcclScalableRootHandle>& rootHandles,
    ScalableServerInfo& scalableInfo);
HcclResult InjectScalableRootMeshInfo(
    const HcclRootHandle& groupRootHandle, bool isRootRank, const ScalableServerInfo& scalableInfo);

#define private public
#define protected public
#include "topoinfo_detect_scalable.h"
#undef protected
#undef private

// 构造 nRoot 个合法的 HcclScalableRootHandle 并塞入 HcclRootInfo.internal，与 HcclGetRootInfoScalable 产出格式一致
static void FillScalableRootInfos(HcclRootInfo* rootInfos, u32 nRoot)
{
    for (u32 i = 0; i < nRoot; ++i) {
        memset(&rootInfos[i], 0, sizeof(HcclRootInfo));
        HcclScalableRootHandle handle{};
        (void)snprintf(handle.rootHandle.identifier, ROOTINFO_INDENTIFIER_MAX_LENGTH, "scalable_root_%u", i);
        handle.rootHandle.port = 5000 + i;
        handle.meshPort = 6000 + i;
        (void)memcpy_s(rootInfos[i].internal, HCCL_ROOT_INFO_BYTES, &handle, sizeof(HcclScalableRootHandle));
    }
}

// ============================ 分组算法（纯静态函数，无外部依赖） ============================
class ScalableGroupAlgorithmTest : public testing::Test {};

// 设计示例：nRanks=11, nRoot=2 → root0 管理 rank0-5（6个），root1 管理 rank6-10（5个）
TEST_F(ScalableGroupAlgorithmTest, Ut_RootIdFromRank_When_NRanks11_NRoot2_Expect_MatchDesign)
{
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 2, 0), 0);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 2, 5), 0);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 2, 6), 1);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 2, 10), 1);
}

TEST_F(ScalableGroupAlgorithmTest, Ut_FirstRankFromRoot_When_NRanks11_NRoot2_Expect_MatchDesign)
{
    EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(11, 2, 0), 0);
    EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(11, 2, 1), 6);
}

TEST_F(ScalableGroupAlgorithmTest, Ut_NRankFromRoot_When_NRanks11_NRoot2_Expect_MatchDesign)
{
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(0, 11, 2), 6);
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(1, 11, 2), 5);
}

// 均分：nRanks=8, nRoot=4 → 每个root各2个rank
TEST_F(ScalableGroupAlgorithmTest, Ut_Grouping_When_EvenDistributed_Expect_MatchDesign)
{
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(8, 4, 0), 0);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(8, 4, 1), 0);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(8, 4, 2), 1);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(8, 4, 7), 3);
    EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(8, 4, 1), 2);
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(0, 8, 4), 2);
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(3, 8, 4), 2);
}

// 多root退化为单root（nRoot=1）：所有rank归属于root0，规模为nRanks
TEST_F(ScalableGroupAlgorithmTest, Ut_Grouping_When_SingleRoot_Expect_AllInRootZero)
{
    const u32 nRanks = 64;
    for (u32 rank = 0; rank < nRanks; ++rank) {
        EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(nRanks, 1, rank), 0);
    }
    EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(nRanks, 1, 0), 0);
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(0, nRanks, 1), nRanks);
}

// 每个rank自成一个root（nRoot=nRanks）：组规模恒为1
TEST_F(ScalableGroupAlgorithmTest, Ut_Grouping_When_EachRankIsRoot_Expect_RankEqualsRoot)
{
    const u32 nRanks = 16;
    for (u32 rank = 0; rank < nRanks; ++rank) {
        EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(nRanks, nRanks, rank), rank);
        EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(nRanks, nRanks, rank), rank);
        EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(rank, nRanks, nRanks), 1);
    }
}

// 不均分：nRanks=11, nRoot=3 → root0/root1各4个，root2为3个
TEST_F(ScalableGroupAlgorithmTest, Ut_RootIdFromRank_When_UnevenDistribution_Expect_MatchDesign)
{
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 3, 0), 0);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 3, 3), 0);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 3, 4), 1);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 3, 7), 1);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 3, 8), 2);
    EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(11, 3, 10), 2);
    EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(11, 3, 0), 0);
    EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(11, 3, 1), 4);
    EXPECT_EQ(TopoInfoDetectScalable::FirstRankFromRoot(11, 3, 2), 8);
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(0, 11, 3), 4);
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(1, 11, 3), 4);
    EXPECT_EQ(TopoInfoDetectScalable::NRankFromRoot(2, 11, 3), 3);
}

// RankHasRoot：rank 是否是其分组的 root（组内第一个 rank）
TEST_F(ScalableGroupAlgorithmTest, Ut_RankHasRoot_When_UnevenDistribution_Expect_MatchDesign)
{
    // nRanks=11, nRoot=2 → root0 管 0-5（root 为 rank0），root1 管 6-10（root 为 rank6）
    EXPECT_TRUE(TopoInfoDetectScalable::RankHasRoot(0, 11, 2));
    EXPECT_TRUE(TopoInfoDetectScalable::RankHasRoot(6, 11, 2));
    EXPECT_FALSE(TopoInfoDetectScalable::RankHasRoot(1, 11, 2));
    EXPECT_FALSE(TopoInfoDetectScalable::RankHasRoot(5, 11, 2));
    EXPECT_FALSE(TopoInfoDetectScalable::RankHasRoot(10, 11, 2));

    // nRanks=11, nRoot=3 → root 为 rank0/4/8
    EXPECT_TRUE(TopoInfoDetectScalable::RankHasRoot(0, 11, 3));
    EXPECT_TRUE(TopoInfoDetectScalable::RankHasRoot(4, 11, 3));
    EXPECT_TRUE(TopoInfoDetectScalable::RankHasRoot(8, 11, 3));
    EXPECT_FALSE(TopoInfoDetectScalable::RankHasRoot(3, 11, 3));
    EXPECT_FALSE(TopoInfoDetectScalable::RankHasRoot(7, 11, 3));
    EXPECT_FALSE(TopoInfoDetectScalable::RankHasRoot(10, 11, 3));

    // 退化场景：nRoot=1 仅 rank0 为 root；nRoot=nRanks 每个 rank 都是 root
    for (u32 rank = 1; rank < 64; ++rank) {
        EXPECT_FALSE(TopoInfoDetectScalable::RankHasRoot(rank, 64, 1));
    }
    for (u32 rank = 0; rank < 16; ++rank) {
        EXPECT_TRUE(TopoInfoDetectScalable::RankHasRoot(rank, 16, 16));
    }
}

// RankHasRoot 与其余分组函数互洽：rank 是 root ⟺ rank == FirstRankFromRoot(RootIdFromRank(rank))
TEST_F(ScalableGroupAlgorithmTest, Ut_RankHasRoot_When_CommonConfigs_Expect_EquivalentToFirstRank)
{
    const u32 configs[][2] = {{1, 1}, {8, 1}, {8, 4}, {11, 2}, {11, 3}, {16, 4}, {100, 7}, {1024, 32}};
    for (auto& cfg : configs) {
        const u32 nRanks = cfg[0];
        const u32 nRoot = cfg[1];
        for (u32 rank = 0; rank < nRanks; ++rank) {
            const u32 rootIdx = TopoInfoDetectScalable::RootIdFromRank(nRanks, nRoot, rank);
            const u32 first = TopoInfoDetectScalable::FirstRankFromRoot(nRanks, nRoot, rootIdx);
            EXPECT_EQ(TopoInfoDetectScalable::RankHasRoot(rank, nRanks, nRoot), (rank == first));
        }
    }
}

// 分组不变量：组内rank连续、互不重叠、覆盖[0, nRanks)，且各组规模之和等于nRanks
TEST_F(ScalableGroupAlgorithmTest, Ut_Grouping_When_CommonConfigs_Expect_PartitionInvariant)
{
    const u32 configs[][2] = {{1, 1}, {8, 1}, {8, 4}, {11, 2}, {11, 3}, {16, 4}, {100, 7}, {1024, 32}};
    for (auto& cfg : configs) {
        const u32 nRanks = cfg[0];
        const u32 nRoot = cfg[1];
        u32 total = 0;
        for (u32 rootIdx = 0; rootIdx < nRoot; ++rootIdx) {
            const u32 first = TopoInfoDetectScalable::FirstRankFromRoot(nRanks, nRoot, rootIdx);
            const u32 cnt = TopoInfoDetectScalable::NRankFromRoot(rootIdx, nRanks, nRoot);
            total += cnt;
            EXPECT_LT(first, nRanks);
            EXPECT_LE(first + cnt, nRanks);
            for (u32 i = 0; i < cnt; ++i) {
                EXPECT_EQ(TopoInfoDetectScalable::RootIdFromRank(nRanks, nRoot, first + i), rootIdx);
            }
        }
        EXPECT_EQ(total, nRanks);
    }
}

// ============================ scalable 建链 API 与内部函数 ============================
class HcclCommInitRootInfoScalableTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // 将建链超时时间设置为1s，减少测试用例运行时间
        MOCKER(GetExternalInputHcclLinkTimeOut).stubs().with(mockcpp::any()).will(returnValue(1));
    }
    void TearDown() override
    {
        // 删除所有拓扑建链的线程
        HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
        opBaseInfo.hcclCommTopoInfoDetectServer.clear();
        opBaseInfo.hcclCommTopoInfoDetectAgent.clear();

        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_NRanksIsZero_Expect_HCCL_E_PARA)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);

    HcclResult ret = HcclCommInitRootInfoScalable(0, 1, nullptr, 0, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_NRootIsZero_Expect_HCCL_E_PARA)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);

    HcclResult ret = HcclCommInitRootInfoScalable(8, 0, nullptr, 0, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_NRootGreaterThanNRanks_Expect_HCCL_E_PARA)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);

    HcclResult ret = HcclCommInitRootInfoScalable(8, 16, nullptr, 0, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_RankGreaterEqualNRanks_Expect_HCCL_E_PARA)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);

    HcclResult ret = HcclCommInitRootInfoScalable(8, 4, nullptr, 8, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_NExtRootNotZero_Expect_HCCL_E_PARA)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);

    HcclResult ret = HcclCommInitRootInfoScalable(8, 4, nullptr, 0, 1, &config, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_CommIsNull_Expect_HCCL_E_PTR)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);

    HcclResult ret = HcclCommInitRootInfoScalable(8, 4, nullptr, 0, 0, &config, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_RootInfoListIsNull_Expect_HCCL_E_PTR)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);

    HcclResult ret = HcclCommInitRootInfoScalable(8, 4, nullptr, 0, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclCommInitRootInfoScalable_When_ConfigIsNull_Expect_HCCL_E_PTR)
{
    Ut_Device_Set(0);
    HcclRootInfo rootInfos[4];
    FillScalableRootInfos(rootInfos, 4);

    HcclResult ret = HcclCommInitRootInfoScalable(8, 4, rootInfos, 0, 0, nullptr, &comm);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_ValidateAndInitRootInfoScalable_When_AllValid_Expect_HCCL_SUCCESS)
{
    Ut_Device_Set(0);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    HcclRootInfo rootInfos[4];
    FillScalableRootInfos(rootInfos, 4);

    HcclResult ret = ValidateAndInitRootInfoScalable(8, 4, 0, 0, &config, &comm, rootInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_ParseScalableRootHandles_When_MultiRoot_Expect_AllParsed)
{
    const u32 nRoot = 3;
    HcclRootInfo rootInfos[nRoot];
    FillScalableRootInfos(rootInfos, nRoot);

    std::vector<HcclScalableRootHandle> rootHandles(nRoot);
    HcclResult ret = ParseScalableRootHandles(rootInfos, nRoot, rootHandles);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(rootHandles.size(), nRoot);
    for (u32 i = 0; i < nRoot; ++i) {
        char expectId[ROOTINFO_INDENTIFIER_MAX_LENGTH];
        (void)snprintf(expectId, ROOTINFO_INDENTIFIER_MAX_LENGTH, "scalable_root_%u", i);
        EXPECT_EQ(rootHandles[i].rootHandle.port, 5000 + i);
        EXPECT_EQ(rootHandles[i].meshPort, 6000 + i);
        EXPECT_STREQ(rootHandles[i].rootHandle.identifier, expectId);
    }
}

// identifier 填满缓冲区且不带结尾'\0'时，解析后需强制截断，避免越界/非终止字符串
TEST_F(HcclCommInitRootInfoScalableTest, Ut_ParseScalableRootHandles_When_IdentifierFull_Expect_Terminated)
{
    HcclRootInfo rootInfo{};
    HcclScalableRootHandle handle{};
    (void)memset(handle.rootHandle.identifier, 'a', ROOTINFO_INDENTIFIER_MAX_LENGTH);
    handle.rootHandle.port = 5000;
    handle.meshPort = 6000;
    (void)memcpy_s(rootInfo.internal, HCCL_ROOT_INFO_BYTES, &handle, sizeof(HcclScalableRootHandle));

    std::vector<HcclScalableRootHandle> rootHandles(1);
    HcclResult ret = ParseScalableRootHandles(&rootInfo, 1, rootHandles);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rootHandles[0].rootHandle.port, 5000);
    EXPECT_EQ(rootHandles[0].meshPort, 6000);
    EXPECT_EQ(rootHandles[0].rootHandle.identifier[0], 'a');
    EXPECT_EQ(rootHandles[0].rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1], '\0');
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_BuildScalableMeshInfo_When_MultiRoot_Expect_MeshInfoCorrect)
{
    const u32 nRoot = 3;
    std::vector<HcclScalableRootHandle> rootHandles(nRoot);
    for (u32 i = 0; i < nRoot; ++i) {
        rootHandles[i].rootHandle.port = 5000 + i;
        rootHandles[i].meshPort = 6000 + i;
        (void)strcpy_s(rootHandles[i].rootHandle.ip, sizeof(rootHandles[i].rootHandle.ip), "127.0.0.1");
    }

    ScalableServerInfo scalableInfo;
    HcclResult ret = BuildScalableMeshInfo(nRoot, 1, 5, rootHandles, scalableInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(scalableInfo.nRoot, nRoot);
    EXPECT_EQ(scalableInfo.rootIndex, 1);
    EXPECT_EQ(scalableInfo.groupSize, 5);
    ASSERT_EQ(scalableInfo.meshInfos.size(), nRoot);
    for (u32 i = 0; i < nRoot; ++i) {
        EXPECT_EQ(scalableInfo.meshInfos[i].meshPort, rootHandles[i].meshPort);
        EXPECT_STREQ(scalableInfo.meshInfos[i].ip, rootHandles[i].rootHandle.ip);
    }
}

TEST_F(HcclCommInitRootInfoScalableTest, Ut_HcclGetRootInfoScalable_When_RootInfoIsNull_Expect_HCCL_E_PTR)
{
    Ut_Device_Set(0);

    HcclResult ret = HcclGetRootInfoScalable(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 注入 mesh 全互联信息后，WaitScalableInfoReady 应立即返回成功
TEST_F(HcclCommInitRootInfoScalableTest, Ut_TopoInfoDetectScalable_SetScalableServerInfo_And_Wait_Expect_Ready)
{
    Ut_Device_Set(0);
    TopoInfoDetectScalable scalableServer;
    ScalableServerInfo info;
    info.nRoot = 4;
    info.rootIndex = 1;
    info.groupSize = 3;

    HcclResult ret = scalableServer.SetScalableServerInfo(info);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(scalableServer.scalableInfoReady_.load());
    EXPECT_EQ(scalableServer.scalableInfo_.nRoot, 4);
    EXPECT_EQ(scalableServer.scalableInfo_.rootIndex, 1);
    EXPECT_EQ(scalableServer.scalableInfo_.groupSize, 3);

    EXPECT_EQ(scalableServer.WaitScalableInfoReady(), HCCL_SUCCESS);
}

// ============================ InjectScalableRootMeshInfo ============================
// 组 root 的 server 尚未登记到 hcclCommTopoInfoDetectServer 时，注入应失败
TEST_F(HcclCommInitRootInfoScalableTest, Ut_InjectScalableRootMeshInfo_When_ServerNotFound_Expect_HCCL_E_INTERNAL)
{
    Ut_Device_Set(0);
    HcclRootHandle groupRootHandle{};
    (void)strcpy_s(groupRootHandle.identifier, sizeof(groupRootHandle.identifier), "not_exist_root");

    ScalableServerInfo scalableInfo;
    HcclResult ret = InjectScalableRootMeshInfo(groupRootHandle, true, scalableInfo);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// 登记的 server 为普通单root TopoInfoDetect（dynamic_cast 失败）时，注入应失败
TEST_F(HcclCommInitRootInfoScalableTest, Ut_InjectScalableRootMeshInfo_When_ServerNotScalable_Expect_HCCL_E_INTERNAL)
{
    Ut_Device_Set(0);
    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    auto plainServer = std::make_shared<TopoInfoDetect>();
    HcclRootHandle groupRootHandle{};
    (void)strcpy_s(groupRootHandle.identifier, sizeof(groupRootHandle.identifier), "plain_root");
    opBaseInfo.hcclCommTopoInfoDetectServer.insert({groupRootHandle.identifier, plainServer});

    ScalableServerInfo scalableInfo;
    HcclResult ret = InjectScalableRootMeshInfo(groupRootHandle, true, scalableInfo);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// 登记的 server 为 TopoInfoDetectScalable 时，注入应成功且 mesh 信息已就绪
TEST_F(HcclCommInitRootInfoScalableTest, Ut_InjectScalableRootMeshInfo_When_ScalableServer_Expect_Success)
{
    Ut_Device_Set(0);
    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    auto scalableServer = std::make_shared<TopoInfoDetectScalable>();
    HcclRootHandle groupRootHandle{};
    (void)strcpy_s(groupRootHandle.identifier, sizeof(groupRootHandle.identifier), "scalable_root_ok");
    opBaseInfo.hcclCommTopoInfoDetectServer.insert({groupRootHandle.identifier, scalableServer});

    ScalableServerInfo scalableInfo;
    scalableInfo.nRoot = 4;
    scalableInfo.rootIndex = 2;
    scalableInfo.groupSize = 8;
    HcclResult ret = InjectScalableRootMeshInfo(groupRootHandle, true, scalableInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(scalableServer->scalableInfoReady_.load());
    EXPECT_EQ(scalableServer->scalableInfo_.nRoot, 4);
    EXPECT_EQ(scalableServer->scalableInfo_.rootIndex, 2);
    EXPECT_EQ(scalableServer->scalableInfo_.groupSize, 8);
}
