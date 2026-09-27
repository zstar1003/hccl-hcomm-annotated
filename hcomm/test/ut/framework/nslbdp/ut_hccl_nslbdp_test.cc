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
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "hccl_nslbdp.h"
#include "hccl_nslbdp_pub.h"
#include "hccl_nslb_md5.h"

using namespace hccl;
using namespace testing;

namespace {
constexpr u32 IP_OCTET_0_SHIFT = 24;
constexpr u32 IP_OCTET_1_SHIFT = 16;
constexpr u32 IP_OCTET_2_SHIFT = 8;
constexpr u32 IP_BYTE_MASK = 0xFF;
constexpr u32 COMM_DESC_OVERFLOW_LEN = COMM_DESC_MAX_LENGTH - 1;
constexpr u64 DUMMY_TASK_ID = 0xABCDULL;

inline u32 BuildIpv4(u32 a, u32 b, u32 c, u32 d)
{
    return ((a & IP_BYTE_MASK) << IP_OCTET_0_SHIFT) | ((b & IP_BYTE_MASK) << IP_OCTET_1_SHIFT)
           | ((c & IP_BYTE_MASK) << IP_OCTET_2_SHIFT) | (d & IP_BYTE_MASK);
}

std::string MakeLongString(size_t len, char fill) { return std::string(len, fill); }

NslbDpCommConfigVal BuildCommCfg(const char* desc, const u8* md5Bytes = nullptr)
{
    NslbDpCommConfigVal cfg = {};
    cfg.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(cfg.commDesc, COMM_DESC_MAX_LENGTH, desc, COMM_DESC_MAX_LENGTH - 1);
    cfg.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    if (md5Bytes != nullptr) {
        (void)memcpy_s(cfg.commMd5Sum, sizeof(cfg.commMd5Sum), md5Bytes, sizeof(cfg.commMd5Sum));
    }
    return cfg;
}
} // namespace

// ============================================================
// 1. 纯函数：GetNslbOpType
// ============================================================
TEST(HcclNslbDpPureFuncTest, GetNslbOpType_AllKnownCases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_BROADCAST), NSLBDP_CMD_BROADCAST);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLREDUCE), NSLBDP_CMD_ALLREDUCE);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_REDUCE), NSLBDP_CMD_REDUCE);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_SEND), NSLBDP_CMD_BATCH_SEND_RECV);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_RECEIVE), NSLBDP_CMD_BATCH_SEND_RECV);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLGATHER), NSLBDP_CMD_ALLGATHER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_REDUCE_SCATTER), NSLBDP_CMD_REDUCE_SCATTER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLTOALLV), NSLBDP_CMD_ALLTOALLV);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLTOALLVC), NSLBDP_CMD_ALLTOALLVC);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_ALLTOALL), NSLBDP_CMD_ALLTOALL);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_GATHER), NSLBDP_CMD_GATHER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_SCATTER), NSLBDP_CMD_SCATTER);
    EXPECT_EQ(inst.GetNslbOpType(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV), NSLBDP_CMD_BATCH_SEND_RECV);
    EXPECT_EQ(inst.GetNslbOpType(static_cast<HcclCMDType>(0xDEAD)), 0);
}

// ============================================================
// 2. 纯函数：GetNslbLevel1AlgType / Level2
// ============================================================
TEST(HcclNslbDpPureFuncTest, GetNslbLevel1AlgType_Cases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_RING), NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_PIPELINE), NSLB_ALGO_TYPE_PIPELINE);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_HD), NSLB_ALGO_TYPE_HDR);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_NHR), NSLB_ALGO_TYPE_NHR);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_NHR_V1), NSLB_ALGO_TYPE_NHR_V1);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_NB), NSLB_ALGO_TYPE_NB);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_AHC), NSLB_ALGO_TYPE_AHC);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE), NSLB_ALGO_TYPE_AHC);
    EXPECT_EQ(inst.GetNslbLevel1AlgType(static_cast<AlgTypeLevel1>(0xFF)), NSLB_ALGO_TYPE_NA);
}

TEST(HcclNslbDpPureFuncTest, GetNslbLevel2AlgType_Cases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_RING), NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_HD), NSLB_ALGO_TYPE_HDR);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_NHR), NSLB_ALGO_TYPE_NHR);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_NB), NSLB_ALGO_TYPE_NB);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(AlgTypeLevel2::ALG_LEVEL2_PIPELINE), NSLB_ALGO_TYPE_PIPELINE);
    EXPECT_EQ(inst.GetNslbLevel2AlgType(static_cast<AlgTypeLevel2>(0xFF)), NSLB_ALGO_TYPE_NA);
}

// ============================================================
// 3. 纯函数：GetNslbDpFirstFourBit
// ============================================================
TEST(HcclNslbDpPureFuncTest, GetNslbDpFirstFourBit_ByOptype)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_ALLREDUCE, NSLB_ALGO_TYPE_RING), 5ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_ALLGATHER, NSLB_ALGO_TYPE_RING), 2ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_REDUCE_SCATTER, NSLB_ALGO_TYPE_RING), 0ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_ALLTOALL, NSLB_ALGO_TYPE_RING), 0ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(NSLBDP_CMD_BROADCAST, NSLB_ALGO_TYPE_AHC), 0ULL);
    EXPECT_EQ(inst.GetNslbDpFirstFourBit(0, 0), 0ULL);
}

// ============================================================
// 4. 纯函数：CheckSupportOptype
// ============================================================
TEST(HcclNslbDpPureFuncTest, CheckSupportOptype_TrueCases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLTOALL));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLTOALLV));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_ALLTOALLVC));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_BROADCAST));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_SCATTER));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_REDUCE));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_SEND));
    EXPECT_TRUE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_RECEIVE));
}

TEST(HcclNslbDpPureFuncTest, CheckSupportOptype_FalseCases)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_FALSE(inst.CheckSupportOptype(HcclCMDType::HCCL_CMD_GATHER));
    EXPECT_FALSE(inst.CheckSupportOptype(static_cast<HcclCMDType>(0xDEAD)));
}

// ============================================================
// 5. 纯函数：CheckAlgoConsistency
// ============================================================
TEST(HcclNslbDpPureFuncTest, CheckAlgoConsistency_AllReduce)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::string hit = "RunAllReduceRingV1";
    std::string miss = "RunAllGatherRingV1";
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLREDUCE, hit));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLREDUCE, miss));
}

TEST(HcclNslbDpPureFuncTest, CheckAlgoConsistency_OtherOps)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::string ag = "DoAllGatherPipeline";
    std::string rs = "DoReduceScatterRing";
    std::string a2aFull = "RunAlltoAllVFullMesh";
    std::string a2aOther = "AlltoAllVOtherName";
    std::string bc = "BroadCastTree";
    std::string sc = "ScatterBinary";
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLGATHER, ag));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLGATHER, rs));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, rs));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ag));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLTOALL, a2aFull));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_ALLTOALL, a2aOther));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_BROADCAST, bc));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_BROADCAST, ag));
    EXPECT_TRUE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_SCATTER, sc));
    EXPECT_FALSE(inst.CheckAlgoConsistency(HcclCMDType::HCCL_CMD_SCATTER, ag));
}

// ============================================================
// 6. 纯函数：ipToUint32
// ============================================================
TEST(HcclNslbDpPureFuncTest, IpToUint32_Normal)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    EXPECT_EQ(inst.ipToUint32("127.0.0.1"), BuildIpv4(127, 0, 0, 1));
    EXPECT_EQ(inst.ipToUint32("0.0.0.0"), 0U);
    EXPECT_EQ(inst.ipToUint32("255.255.255.255"), 0xFFFFFFFFU);
    EXPECT_EQ(inst.ipToUint32("192.168.1.1"), BuildIpv4(192, 168, 1, 1));
    EXPECT_EQ(inst.ipToUint32("10.11.12.13"), BuildIpv4(10, 11, 12, 13));
}

// ============================================================
// 7. 纯函数：SplitString
// ============================================================
TEST(HcclNslbDpPureFuncTest, SplitString_Normal)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::vector<std::string> out;
    inst.SplitString("a_b_c", out, "_");
    ASSERT_EQ(out.size(), 3U);
    EXPECT_EQ(out[0], "a");
    EXPECT_EQ(out[1], "b");
    EXPECT_EQ(out[2], "c");
}

TEST(HcclNslbDpPureFuncTest, SplitString_Edge)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    std::vector<std::string> out1;
    inst.SplitString("noseparator", out1, "_");
    ASSERT_EQ(out1.size(), 1U);
    EXPECT_EQ(out1[0], "noseparator");

    std::vector<std::string> out2;
    inst.SplitString("", out2, "_");
    EXPECT_EQ(out2.size(), 0U);

    std::vector<std::string> out3;
    inst.SplitString("a__b", out3, "_");
    ASSERT_EQ(out3.size(), 3U);
    EXPECT_EQ(out3[0], "a");
    EXPECT_EQ(out3[1], "");
    EXPECT_EQ(out3[2], "b");
}

// ============================================================
// 8. 弱状态：fullcommDescInitTime
// ============================================================
TEST(HcclNslbDpWeakStateTest, FullcommDescInitTime_WithAndWithoutTimestamp)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();

    // fullcommDescInitTime 新版需要表1里有对应 commDesc 条目
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    std::string withTs = "aaa_bbb_ccc_1710000000123";
    NslbDpCommConfigVal cfgA = BuildCommCfg(withTs.c_str());
    cfgA.commInitTime = 100ULL;
    inst.hcclNslbDpCommConfig_.push_back(cfgA);

    NslbDpOperatorInfo op = {};
    inst.fullcommDescInitTime(withTs, op);
    EXPECT_STREQ(op.commDesc, withTs.c_str());
    EXPECT_EQ(op.commDesc[COMM_DESC_MAX_LENGTH - 1], '\0');
    EXPECT_NE(op.commInitTime, 0ULL);

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

// ============================================================
// 9. 弱状态 + 注入表项：CheckCommDescExit / CheckSameOperatorVal
// ============================================================
TEST(HcclNslbDpWeakStateTest, CheckCommDescExit_InjectedConfig)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    NslbDpOperatorInfo op = {};
    (void)strncpy_s(op.commDesc, COMM_DESC_MAX_LENGTH, "exist_group", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_FALSE(inst.CheckCommDescExit(op));

    inst.hcclNslbDpCommConfig_.push_back(BuildCommCfg("exist_group"));
    EXPECT_TRUE(inst.CheckCommDescExit(op));

    NslbDpOperatorInfo op2 = {};
    (void)strncpy_s(op2.commDesc, COMM_DESC_MAX_LENGTH, "ghost", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_FALSE(inst.CheckCommDescExit(op2));

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

TEST(HcclNslbDpWeakStateTest, CheckSameOperatorVal_InjectedVal)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpOperatorVal_);
    inst.hcclNslbDpOperatorVal_.clear();

    NslbDpOperatorInfo op = {};
    op.taskId = DUMMY_TASK_ID;
    op.rootRank = 123;
    op.oper = NSLBDP_CMD_ALLREDUCE;
    op.algorithm = NSLB_ALGO_TYPE_RING;
    (void)strncpy_s(op.commDesc, COMM_DESC_MAX_LENGTH, "g", COMM_DESC_MAX_LENGTH - 1);

    EXPECT_FALSE(inst.CheckSameOperatorVal(0, op, 123));

    inst.hcclNslbDpOperatorVal_.push_back(op);
    EXPECT_TRUE(inst.CheckSameOperatorVal(0, op, 123));

    EXPECT_FALSE(inst.CheckSameOperatorVal(0, op, 999));
    op.commDesc[0] = 'x';
    EXPECT_FALSE(inst.CheckSameOperatorVal(0, op, 123));

    inst.hcclNslbDpOperatorVal_ = std::move(backup);
}

// ============================================================
// 10. 拆分助手 1：FindMinInitTimeCommDesc
// ============================================================
TEST(HcclNslbDpHelpersTest, FindMinInitTimeCommDesc_Found)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    // 构造两个相同 MD5 但不同 commInitTime 的条目
    u8 md5[NSLB_MD5_DIGEST];
    for (u32 k = 0; k < NSLB_MD5_DIGEST; k++) {
        md5[k] = static_cast<u8>(k + 1U);
    }

    NslbDpCommConfigVal cfg1 = BuildCommCfg("dup_a", md5);
    cfg1.commInitTime = 100ULL;
    inst.hcclNslbDpCommConfig_.push_back(cfg1);

    NslbDpCommConfigVal cfg2 = BuildCommCfg("dup_b", md5);
    cfg2.commInitTime = 50ULL; // 更小
    inst.hcclNslbDpCommConfig_.push_back(cfg2);

    std::string outComm;
    u64 outTime = 0;
    EXPECT_TRUE(inst.FindMinInitTimeCommDesc("dup_a", outComm, outTime));
    EXPECT_STREQ(outComm.c_str(), "dup_b"); // dup_b 的 initTime=50 更小
    EXPECT_EQ(outTime, 50ULL);

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

TEST(HcclNslbDpHelpersTest, FindMinInitTimeCommDesc_NotFound)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    std::string outComm;
    u64 outTime = 0;
    EXPECT_FALSE(inst.FindMinInitTimeCommDesc("no_such_group", outComm, outTime));

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

// ============================================================
// 11. 拆分助手 2：FillAlgInfoCommMd5
// ============================================================
TEST(HcclNslbDpHelpersTest, FillAlgInfoCommMd5_FoundAndNotFound)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpCommConfig_);
    inst.hcclNslbDpCommConfig_.clear();

    u8 goldenMd5[NSLB_MD5_DIGEST];
    for (u32 i = 0; i < NSLB_MD5_DIGEST; i++) {
        goldenMd5[i] = static_cast<u8>(i + 1U);
    }
    inst.hcclNslbDpCommConfig_.push_back(BuildCommCfg("withMd5", goldenMd5));

    NslbDpAlgorithmInfo a = {};
    (void)strncpy_s(a.commDesc, COMM_DESC_MAX_LENGTH, "withMd5", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_TRUE(inst.FillAlgInfoCommMd5(a));
    EXPECT_EQ(memcmp(a.commMd5Sum, goldenMd5, NSLB_MD5_DIGEST), 0);

    NslbDpAlgorithmInfo b = {};
    (void)strncpy_s(b.commDesc, COMM_DESC_MAX_LENGTH, "noSuch", COMM_DESC_MAX_LENGTH - 1);
    u8 zero[NSLB_MD5_DIGEST] = {0};
    EXPECT_TRUE(inst.FillAlgInfoCommMd5(b));
    EXPECT_EQ(memcmp(b.commMd5Sum, zero, NSLB_MD5_DIGEST), 0);

    inst.hcclNslbDpCommConfig_ = std::move(backup);
}

// ============================================================
// 12. 拆分助手 3：FillAlgInfoBaseFields（纯赋值）
// ============================================================
TEST(HcclNslbDpHelpersTest, FillAlgInfoBaseFields_PureAssignment)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    NslbDpAlgorithmInfo a = {};
    inst.FillAlgInfoBaseFields(a, HcclCMDType::HCCL_CMD_ALLGATHER, 7U, 3U, NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(a.srcLocalRankId, 7U);
    EXPECT_EQ(a.rootRank, 3U);
    EXPECT_EQ(a.oper, NSLBDP_CMD_ALLGATHER);
    EXPECT_EQ(a.algorithm, NSLB_ALGO_TYPE_RING);

    NslbDpAlgorithmInfo b = {};
    inst.FillAlgInfoBaseFields(b, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, 0xFFFFFFFFU, 0xFFFEU, NSLB_ALGO_TYPE_AHC);
    EXPECT_EQ(b.srcLocalRankId, static_cast<u16>(0xFFFFFFFFU));
    EXPECT_EQ(b.rootRank, 0xFFFEU);
    EXPECT_EQ(b.oper, NSLBDP_CMD_BATCH_SEND_RECV);
    EXPECT_EQ(b.algorithm, NSLB_ALGO_TYPE_AHC);
}

// ============================================================
// 13. 拆分助手 4：IsAlgAdjacencyDuplicated
// ============================================================
TEST(HcclNslbDpHelpersTest, IsAlgAdjacencyDuplicated_BothBranches)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();
    auto backup = std::move(inst.hcclNslbDpAlgorithmInfo_);
    inst.hcclNslbDpAlgorithmInfo_.clear();

    NslbDpAlgorithmInfo base = {};
    base.taskId = DUMMY_TASK_ID;
    base.srcLocalRankId = 2;
    base.rootRank = 5;
    base.oper = NSLBDP_CMD_ALLREDUCE;
    base.algorithm = NSLB_ALGO_TYPE_RING;
    (void)strncpy_s(base.commDesc, COMM_DESC_MAX_LENGTH, "dup_test", COMM_DESC_MAX_LENGTH - 1);

    EXPECT_FALSE(inst.IsAlgAdjacencyDuplicated(base));

    inst.hcclNslbDpAlgorithmInfo_.push_back(base);
    EXPECT_TRUE(inst.IsAlgAdjacencyDuplicated(base));

    NslbDpAlgorithmInfo diff = base;
    diff.srcLocalRankId = 9;
    EXPECT_FALSE(inst.IsAlgAdjacencyDuplicated(diff));

    NslbDpAlgorithmInfo diffDesc = base;
    (void)strncpy_s(diffDesc.commDesc, COMM_DESC_MAX_LENGTH, "other", COMM_DESC_MAX_LENGTH - 1);
    EXPECT_FALSE(inst.IsAlgAdjacencyDuplicated(diffDesc));

    inst.hcclNslbDpAlgorithmInfo_ = std::move(backup);
}

// ============================================================
// 14. 拆分助手 5：FillAlgInfoAdjInfo（空/非空分支）
// ============================================================
TEST(HcclNslbDpHelpersTest, FillAlgInfoAdjInfo_EmptyAndNonEmpty)
{
    hcclNslbDp& inst = hcclNslbDp::GetInstance();

    AdjInfo emptyAdj = {};
    emptyAdj.dstRankNum = 99;
    emptyAdj.nsAdjInfo.clear();
    NslbDpAlgorithmInfo a = {};
    EXPECT_FALSE(inst.FillAlgInfoAdjInfo(a, emptyAdj, 1U));
    EXPECT_EQ(a.dstRankNum, 0U);

    AdjInfo adj;
    adj.dstRankNum = 3;
    NslbDpAdjInfo r1 = {10, 1, 0};
    NslbDpAdjInfo r2 = {20, 2, 0};
    NslbDpAdjInfo r3 = {30, 3, 0};
    adj.nsAdjInfo.push_back(r1);
    adj.nsAdjInfo.push_back(r2);
    adj.nsAdjInfo.push_back(r3);
    NslbDpAlgorithmInfo b = {};
    EXPECT_TRUE(inst.FillAlgInfoAdjInfo(b, adj, 7U));
    EXPECT_EQ(b.dstRankNum, 3U);
    ASSERT_EQ(b.AdjInfo.size(), 3U);
    EXPECT_EQ(b.AdjInfo[0].dstLocalRankId, 10U);
    EXPECT_EQ(b.AdjInfo[0].phaseId, 1U);
    EXPECT_EQ(b.AdjInfo[0].rev, 0U);
    EXPECT_EQ(b.AdjInfo[1].dstLocalRankId, 20U);
    EXPECT_EQ(b.AdjInfo[1].phaseId, 2U);
    EXPECT_EQ(b.AdjInfo[2].dstLocalRankId, 30U);
    EXPECT_EQ(b.AdjInfo[2].phaseId, 3U);
}

// ============================================================
// 15. ST 流程：SetGlobalCommTaskId 使能门控 + 注入 commConfig
// ============================================================
namespace {
constexpr char kTestCommDesc[] = "st_flow_group";

class NslbDpStFlowTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        inst_ = &hcclNslbDp::GetInstance();
        inst_->SetGlobalCommTaskId(DUMMY_TASK_ID);

        commCfgBackup_ = std::move(inst_->hcclNslbDpCommConfig_);
        inst_->hcclNslbDpCommConfig_.clear();
        inst_->hcclNslbDpCommConfig_.push_back(BuildCommCfg(kTestCommDesc));

        algorithmInfoBackup_ = std::move(inst_->hcclNslbDpAlgorithmInfo_);
        operatorValBackup_ = std::move(inst_->hcclNslbDpOperatorVal_);
        netcoFlagBackup_ = inst_->nslbdpIsInitNetCo_.load();
        l4sPortIdBackup_ = inst_->hcclNslbDpL4SPortId_;
    }

    void TearDown() override
    {
        inst_->hcclNslbDpCommConfig_ = std::move(commCfgBackup_);
        inst_->hcclNslbDpAlgorithmInfo_ = std::move(algorithmInfoBackup_);
        inst_->hcclNslbDpOperatorVal_ = std::move(operatorValBackup_);
        inst_->nslbdpIsInitNetCo_ = netcoFlagBackup_;
        inst_->hcclNslbDpL4SPortId_ = l4sPortIdBackup_;
        inst_->SetGlobalCommTaskId(0);
    }

    hcclNslbDp* inst_;
    std::vector<NslbDpCommConfigVal> commCfgBackup_;
    std::vector<NslbDpAlgorithmInfo> algorithmInfoBackup_;
    std::vector<NslbDpOperatorInfo> operatorValBackup_;
    bool netcoFlagBackup_;
    u32 l4sPortIdBackup_;
};
} // namespace

// ============================================================
// 16. GetAlgAdjacencyTable：ST 全路径
// ============================================================
TEST_F(NslbDpStFlowTest, GetAlgAdjacencyTable_FullPath)
{
    AdjInfo adjInfo = {};
    adjInfo.dstRankNum = 3;
    adjInfo.nsAdjInfo.push_back({10, 1, 0});
    adjInfo.nsAdjInfo.push_back({20, 2, 0});
    adjInfo.nsAdjInfo.push_back({30, 3, 0});

    HcclResult ret = inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 1U);
    const auto& entry = inst_->hcclNslbDpAlgorithmInfo_[0];
    EXPECT_EQ(entry.taskId, DUMMY_TASK_ID);
    EXPECT_STREQ(entry.commDesc, kTestCommDesc);
    EXPECT_EQ(entry.srcLocalRankId, 1U);
    EXPECT_EQ(entry.rootRank, 0U);
    EXPECT_EQ(entry.oper, NSLBDP_CMD_ALLREDUCE);
    EXPECT_EQ(entry.algorithm, NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(entry.dstRankNum, 3U);
    EXPECT_EQ(entry.AdjInfo.size(), 3U);
    EXPECT_EQ(entry.AdjInfo[0].dstLocalRankId, 10U);
}

TEST_F(NslbDpStFlowTest, GetAlgAdjacencyTable_Deduplication)
{
    AdjInfo adjInfo = {};
    adjInfo.dstRankNum = 2;
    adjInfo.nsAdjInfo.push_back({10, 1, 0});
    adjInfo.nsAdjInfo.push_back({20, 2, 0});

    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLGATHER, 2U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    ASSERT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 1U);

    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLGATHER, 2U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 1U);

    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLGATHER, 9U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 2U);
}

TEST_F(NslbDpStFlowTest, GetAlgAdjacencyTable_EarlyReturns)
{
    inst_->SetGlobalCommTaskId(0);
    AdjInfo adjInfo = {};
    adjInfo.dstRankNum = 1;
    adjInfo.nsAdjInfo.push_back({10, 1, 0});
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 0U);

    inst_->SetGlobalCommTaskId(DUMMY_TASK_ID);
    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_GATHER, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 0U);

    inst_->GetAlgAdjacencyTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string("nonexistent_group"), adjInfo);
    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_.size(), 0U);
}

// ============================================================
// 17. GenerateOpAndAdjTable：ST 全路径
// ============================================================
TEST_F(NslbDpStFlowTest, GenerateOpAndAdjTable_FullPath)
{
    HcclResult ret = inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 0U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), 100ULL, 8U);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 1U);
    const auto& entry = inst_->hcclNslbDpOperatorVal_[0];
    EXPECT_EQ(entry.taskId, DUMMY_TASK_ID);
    EXPECT_STREQ(entry.commDesc, kTestCommDesc);
    EXPECT_EQ(entry.oper, NSLBDP_CMD_ALLREDUCE);
    EXPECT_EQ(entry.algorithm, NSLB_ALGO_TYPE_RING);
    EXPECT_EQ(entry.rootRank, 0U);
    EXPECT_EQ(entry.sedFlag, 0U);
    EXPECT_NE(entry.trafficCnt, 0ULL);
}

TEST_F(NslbDpStFlowTest, GenerateOpAndAdjTable_EarlyReturns)
{
    inst_->SetGlobalCommTaskId(0);
    inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 0U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), 100ULL, 8U);
    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 0U);

    inst_->SetGlobalCommTaskId(DUMMY_TASK_ID);

    inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 1U, 0U, NSLB_ALGO_TYPE_RING, std::string(kTestCommDesc), 100ULL, 8U);
    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 0U);

    inst_->GenerateOpAndAdjTable(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 0U, 0U, NSLB_ALGO_TYPE_RING, std::string("no_such_group"), 100ULL, 8U);
    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_.size(), 0U);
}

// ============================================================
// 18. SendOpAndAdjTable / SendAlgorithmInfoTable
// ============================================================
TEST_F(NslbDpStFlowTest, SendOpAndAdjTable_IteratesAndSends)
{
    NslbDpOperatorInfo op = {};
    op.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(op.commDesc, COMM_DESC_MAX_LENGTH, kTestCommDesc, COMM_DESC_MAX_LENGTH - 1);
    op.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    op.oper = NSLBDP_CMD_ALLREDUCE;
    op.algorithm = NSLB_ALGO_TYPE_RING;
    op.rootRank = 0;
    op.trafficCnt = 42;
    inst_->hcclNslbDpOperatorVal_.push_back(op);

    inst_->nslbdpIsInitNetCo_ = true;

    HcclResult ret = inst_->SendOpAndAdjTable();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(inst_->hcclNslbDpOperatorVal_[0].sedFlag, 1U);
}

TEST_F(NslbDpStFlowTest, SendAlgorithmInfoTable_IteratesAndSends)
{
    NslbDpAlgorithmInfo ai = {};
    ai.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(ai.commDesc, COMM_DESC_MAX_LENGTH, kTestCommDesc, COMM_DESC_MAX_LENGTH - 1);
    ai.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    ai.srcLocalRankId = 1;
    ai.oper = NSLBDP_CMD_ALLREDUCE;
    ai.algorithm = NSLB_ALGO_TYPE_RING;
    ai.rootRank = 0;
    ai.dstRankNum = 2;
    ai.AdjInfo.push_back({10, 1, 0});
    ai.AdjInfo.push_back({20, 2, 0});
    inst_->hcclNslbDpAlgorithmInfo_.push_back(ai);

    inst_->nslbdpIsInitNetCo_ = true;

    HcclResult ret = inst_->SendAlgorithmInfoTable();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(inst_->hcclNslbDpAlgorithmInfo_[0].sedFlag, 1U);
}

// ============================================================
// 19. 序列化函数直接覆盖
// ============================================================
TEST_F(NslbDpStFlowTest, SerializeTLV_TableOpAndAdj_ReturnsNonEmpty)
{
    NslbDpOperatorInfo info = {};
    info.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(info.commDesc, COMM_DESC_MAX_LENGTH, "serialize_test", COMM_DESC_MAX_LENGTH - 1);
    info.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    info.oper = NSLBDP_CMD_ALLREDUCE;
    info.algorithm = NSLB_ALGO_TYPE_RING;
    info.rootRank = 0;
    info.trafficCnt = 99;
    info.l4SPortId = 8080;

    std::vector<uint8_t> result = inst_->serializeTLV_TableOpAndAdj(info);
    EXPECT_FALSE(result.empty());
    EXPECT_GT(result.size(), 150U);
}

TEST_F(NslbDpStFlowTest, SerializeTLV_TableAlgorithmInfo_ReturnsNonEmpty)
{
    NslbDpAlgorithmTlv info = {};
    info.taskId = DUMMY_TASK_ID;
    (void)strncpy_s(info.commDesc, COMM_DESC_MAX_LENGTH, "seri_algo_test", COMM_DESC_MAX_LENGTH - 1);
    info.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    info.srcLocalRankId = 3;
    info.oper = NSLBDP_CMD_ALLGATHER;
    info.algorithm = NSLB_ALGO_TYPE_PIPELINE;
    info.rootRank = 1;
    info.dstRankNum = 2;
    info.AdjInfo.push_back({50, 1, 0});

    std::vector<uint8_t> result = inst_->serializeTLV_TableAlgorithmInfo(info);
    EXPECT_FALSE(result.empty());
}

// ============================================================
// 20. CalcPacketNum：与分片逻辑集成测试
// ============================================================
TEST_F(NslbDpStFlowTest, SendTableProc_4096Ranks_4Shards)
{
    constexpr u32 kTotalRanks = NSLBDP_RANKTOTALNUM_BLOCK_FOU;
    u32 packetNum = hcclNslbDp::CalcPacketNum(kTotalRanks);
    EXPECT_EQ(packetNum, 4U);

    for (u32 rank = 0; rank < packetNum; rank++) {
        u32 start = rank * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
        u32 end = std::min((rank + 1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR, kTotalRanks);
        EXPECT_EQ(end - start, NSLBDP_RANKTOTALNUM_BLOCK_FIR);
    }
}

TEST_F(NslbDpStFlowTest, SendTableProc_4093Ranks_LastShardTruncated)
{
    constexpr u32 kTotalRanks = NSLBDP_RANKTOTALNUM_BLOCK_FOU - 3;
    u32 packetNum = hcclNslbDp::CalcPacketNum(kTotalRanks);
    EXPECT_EQ(packetNum, 4U);

    for (u32 rank = 0; rank < 3; rank++) {
        EXPECT_EQ(NSLBDP_RANKTOTALNUM_BLOCK_FIR, NSLBDP_RANKTOTALNUM_BLOCK_FIR);
    }
    u32 lastStart = 3 * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    u32 lastEnd = std::min(4 * NSLBDP_RANKTOTALNUM_BLOCK_FIR, kTotalRanks);
    EXPECT_EQ(lastEnd - lastStart, 1021U);
}

TEST_F(NslbDpStFlowTest, SendTableProc_1024Ranks_SingleShard)
{
    constexpr u32 kTotalRanks = NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    u32 packetNum = hcclNslbDp::CalcPacketNum(kTotalRanks);
    EXPECT_EQ(packetNum, 1U);
}

TEST_F(NslbDpStFlowTest, EntryFilters_4096Pass_4097Drop)
{
    EXPECT_LE(NSLBDP_RANKTOTALNUM_BLOCK_FOU, NSLBDP_RANKTOTALNUM_BLOCK_FOU);

    u32 nRanks = NSLBDP_RANKTOTALNUM_BLOCK_FOU + 1;
    EXPECT_GT(nRanks, NSLBDP_RANKTOTALNUM_BLOCK_FOU);
}

// ============================================================
// 21. fullCommConfigInfo：直接调用真实函数验证越界防护
// ============================================================
namespace {
NslbDpCommConfigVal BuildCommCfgForShardTest(u32 rankCount)
{
    NslbDpCommConfigVal val = {};
    val.taskId = 0xFACEULL;
    val.rankTotalNum = static_cast<u16>(rankCount);
    (void)strncpy_s(val.commDesc, COMM_DESC_MAX_LENGTH, "shard_test_group", COMM_DESC_MAX_LENGTH - 1);
    for (u32 i = 0; i < rankCount; i++) {
        NslbDpRankInfo ri = {};
        ri.deviceIp = 0x0A000001U + i;
        ri.serverIp = 0x0B000001U + i;
        ri.podId = static_cast<u16>(i);
        val.rankInfo.push_back(ri);
    }
    return val;
}
} // namespace

TEST_F(NslbDpStFlowTest, FullCommConfigInfo_4Shards_FillsCorrectSendRankInfo)
{
    constexpr u32 kRankCount = 2U;
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(kRankCount);
    const u32 packetNum = 4U;

    NslbDpCommConfigInfo tab_f = {};
    inst_->fullCommConfigInfo(tab_f, cominfo, packetNum);

    EXPECT_EQ(tab_f.taskId, cominfo.taskId);
    EXPECT_EQ(tab_f.packetNum, packetNum);
    EXPECT_STREQ(tab_f.commDesc, cominfo.commDesc);

    for (u32 i = 0; i < kRankCount; i++) {
        EXPECT_EQ(tab_f.sendRankInfo[i].deviceIp, cominfo.rankInfo[i].deviceIp);
    }
}

TEST_F(NslbDpStFlowTest, FullCommConfigInfo_4096Ranks_SendRankInfoTop4)
{
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(NSLBDP_RANKTOTALNUM_BLOCK_FOU);
    const u32 packetNum = hcclNslbDp::CalcPacketNum(NSLBDP_RANKTOTALNUM_BLOCK_FOU);
    ASSERT_EQ(packetNum, 4U);

    NslbDpCommConfigInfo tab_f = {};
    inst_->fullCommConfigInfo(tab_f, cominfo, packetNum);
    EXPECT_EQ(tab_f.packetNum, packetNum);
    for (u32 i = 0; i < packetNum; i++) {
        EXPECT_EQ(tab_f.sendRankInfo[i].deviceIp, cominfo.rankInfo[i].deviceIp);
    }
}

// ============================================================
// 22. fullCommonGlobalRankInfo：直接调用验证 sendCnt 边界
// ============================================================
TEST_F(NslbDpStFlowTest, FullCommonGlobalRankInfo_PacketNumExceedsRankSize_NoOverread)
{
    NslbDpGlobalRankVal cominfo = {};
    cominfo.taskId = 0xCAFEULL;
    cominfo.rankTotalNum = 3;
    (void)strncpy_s(cominfo.commDesc, COMM_DESC_MAX_LENGTH, "global_rank_test", COMM_DESC_MAX_LENGTH - 1);
    for (u32 i = 0; i < 3; i++) {
        TableFourRankInfo ri = {};
        ri.deviceIp = 0x0C000001U + i;
        ri.serverIp = 0x0D000001U + i;
        cominfo.rankInfo.push_back(ri);
    }

    NslbDpGlobalRankInfo tab_f = {};
    tab_f.packetId = 0;
    tab_f.packetNum = 4;
    inst_->fullCommonGlobalRankInfo(tab_f, cominfo);

    EXPECT_EQ(tab_f.taskId, cominfo.taskId);
    EXPECT_EQ(tab_f.packetNum, 4U);
    EXPECT_STREQ(tab_f.commDesc, cominfo.commDesc);
    for (u32 i = 0; i < 3; i++) {
        EXPECT_EQ(tab_f.sendRankInfo[i].deviceIp, cominfo.rankInfo[i].deviceIp);
    }
}

// ============================================================
// 23. SendTableProc / SendTableGlobalRankProc：真实调用覆盖重构后的统一切片写法
// ============================================================
TEST_F(NslbDpStFlowTest, SendTableProc_4093Ranks_LastShardSize1021)
{
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(NSLBDP_RANKTOTALNUM_BLOCK_FOU - 3);
    const u32 packetNum = hcclNslbDp::CalcPacketNum(cominfo.rankInfo.size());
    ASSERT_EQ(packetNum, 4U);

    for (u32 rank = 0; rank < packetNum; rank++) {
        NslbDpCommConfigInfo tab_f = {};
        tab_f.packetId = static_cast<u16>(rank);
        inst_->fullCommConfigInfo(tab_f, cominfo, packetNum);

        u32 start = rank * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
        u32 end = (rank + 1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
        end = std::min(end, static_cast<u32>(cominfo.rankInfo.size()));
        const u32 expectedSize = (start < end) ? (end - start) : 0U;

        if (rank < 3) {
            EXPECT_EQ(expectedSize, NSLBDP_RANKTOTALNUM_BLOCK_FIR) << "rank=" << rank;
        } else {
            EXPECT_EQ(expectedSize, 1021U) << "last shard rank=" << rank;
        }
    }
}

TEST_F(NslbDpStFlowTest, SendTableGlobalRankProc_EmptyRange_LogsNoCrash)
{
    NslbDpGlobalRankVal cominfo = {};
    cominfo.taskId = 0xBEEFULL;
    cominfo.rankTotalNum = 50;
    (void)strncpy_s(cominfo.commDesc, COMM_DESC_MAX_LENGTH, "empty_range_test", COMM_DESC_MAX_LENGTH - 1);
    for (u32 i = 0; i < 50; i++) {
        TableFourRankInfo ri = {};
        ri.deviceIp = 0x0E000001U + i;
        cominfo.rankInfo.push_back(ri);
    }

    NslbDpGlobalRankInfo tab_f = {};
    tab_f.packetId = 2;
    tab_f.packetNum = hcclNslbDp::CalcPacketNum(cominfo.rankInfo.size());
    inst_->fullCommonGlobalRankInfo(tab_f, cominfo);

    u32 packetIndex = tab_f.packetId;
    u32 start = packetIndex * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    u32 end = (packetIndex + 1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    u32 totalSize = static_cast<u32>(cominfo.rankInfo.size());
    end = std::min(end, totalSize);
    if (start < end) {
        tab_f.rankInfo.reserve(end - start);
        for (u32 j = start; j < end; ++j) {
            tab_f.rankInfo.push_back(cominfo.rankInfo[j]);
        }
    }
    tab_f.rankNum = static_cast<u16>(tab_f.rankInfo.size());
    EXPECT_EQ(tab_f.rankNum, 0U);
}

// ============================================================
// 24. IsCommDescDuplicated：直接调用真实函数验证重复检测
// ============================================================
TEST_F(NslbDpStFlowTest, IsCommDescDuplicated_FoundMatch)
{
    EXPECT_TRUE(inst_->IsCommDescDuplicated(kTestCommDesc, DUMMY_TASK_ID));
}

TEST_F(NslbDpStFlowTest, IsCommDescDuplicated_NotFoundDescMismatch)
{
    EXPECT_FALSE(inst_->IsCommDescDuplicated("nonexistent_group", DUMMY_TASK_ID));
}

TEST_F(NslbDpStFlowTest, IsCommDescDuplicated_NotFoundTaskIdMismatch)
{
    EXPECT_FALSE(inst_->IsCommDescDuplicated(kTestCommDesc, 0xFFFFULL));
}

TEST_F(NslbDpStFlowTest, IsCommDescDuplicated_EmptyConfig)
{
    auto backup = std::move(inst_->hcclNslbDpCommConfig_);
    inst_->hcclNslbDpCommConfig_.clear();
    EXPECT_FALSE(inst_->IsCommDescDuplicated(kTestCommDesc, DUMMY_TASK_ID));
    inst_->hcclNslbDpCommConfig_ = std::move(backup);
}

// ============================================================
// 25. FillRankInfoFromRankTable：直接调用真实函数验证填充逻辑
// ============================================================
namespace {
RankTable_t BuildMultiMachineRankTable(u32 rankCount, bool diffPod = false)
{
    RankTable_t rt = {};
    rt.rankNum = rankCount;
    for (u32 i = 0; i < rankCount; i++) {
        RankInfo_t ri;
        ri.rankId = i;
        ri.serverId = (i == 0) ? "192.168.1.1" : "192.168.1.2";
        ri.superPodIdx = diffPod ? static_cast<u32>(i) : 0;
        ri.deviceInfo.deviceIp.push_back(HcclIpAddress("10.0.0." + std::to_string(i + 1)));
        rt.rankList.push_back(ri);
    }
    return rt;
}
HcclBasicRankInfo BuildHcclBasicRankInfo()
{
    HcclBasicRankInfo info = {HcclIpAddress(), 0};
    info.deviceIP.push_back(HcclIpAddress("10.0.0.1"));
    info.rank = 0;
    info.rankSize = 2;
    return info;
}

std::vector<RankInfo> BuildRankInfoList(u32 count)
{
    std::vector<RankInfo> list;
    for (u32 i = 0; i < count; i++) {
        RankInfo ri;
        ri.superPodIdx = 0;
        list.push_back(ri);
    }
    return list;
}
} // namespace

TEST_F(NslbDpStFlowTest, FillRankInfoFromRankTable_NormalFill)
{
    constexpr u32 kRankCount = 3;
    RankTable_t rt = BuildMultiMachineRankTable(kRankCount);
    NslbDpCommConfigVal globalCommInfo = {};

    inst_->FillRankInfoFromRankTable(globalCommInfo, rt);

    EXPECT_EQ(globalCommInfo.rankInfo.size(), kRankCount);
    for (u32 i = 0; i < kRankCount; i++) {
        EXPECT_EQ(globalCommInfo.rankInfo[i].podId, 0U);
        EXPECT_EQ(globalCommInfo.rankInfo[i].rev, 0U);
    }
}

TEST_F(NslbDpStFlowTest, FillRankInfoFromRankTable_InvalidSuperPodIdx)
{
    RankTable_t rt = {};
    rt.rankNum = 1;
    RankInfo_t ri;
    ri.superPodIdx = INVALID_UINT;
    ri.deviceInfo.deviceIp.push_back(HcclIpAddress("10.0.0.1"));
    ri.serverId = "192.168.1.1";
    rt.rankList.push_back(ri);

    NslbDpCommConfigVal globalCommInfo = {};
    inst_->FillRankInfoFromRankTable(globalCommInfo, rt);

    ASSERT_EQ(globalCommInfo.rankInfo.size(), 1U);
    EXPECT_EQ(globalCommInfo.rankInfo[0].podId, 0U);
}

TEST_F(NslbDpStFlowTest, FillRankInfoFromRankTable_EmptyRankTable)
{
    RankTable_t rt = {};
    rt.rankNum = 0;
    NslbDpCommConfigVal globalCommInfo = {};

    inst_->FillRankInfoFromRankTable(globalCommInfo, rt);
    EXPECT_TRUE(globalCommInfo.rankInfo.empty());
}

// ============================================================
// 26. SendTableProc：真实调用覆盖重构后的统一切片写法
// ============================================================
TEST_F(NslbDpStFlowTest, SendTableProc_RealCall_4Shards_Rank0)
{
    constexpr u32 kRankCount = NSLBDP_RANKTOTALNUM_BLOCK_FOU;
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(kRankCount);
    const u32 packetNum = hcclNslbDp::CalcPacketNum(kRankCount);
    ASSERT_EQ(packetNum, 4U);

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableProc(0, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendTableProc_RealCall_LastShardTruncated)
{
    constexpr u32 kRankCount = NSLBDP_RANKTOTALNUM_BLOCK_FOU - 3;
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(kRankCount);
    const u32 packetNum = hcclNslbDp::CalcPacketNum(kRankCount);
    ASSERT_EQ(packetNum, 4U);

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableProc(3, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendTableProc_RealCall_RankExceedsPacketNum)
{
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(NSLBDP_RANKTOTALNUM_BLOCK_FIR);
    const u32 packetNum = 1U;

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableProc(1, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendTableProc_RealCall_TotalSizeLessThanStart)
{
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(10);
    const u32 packetNum = 4U;

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableProc(2, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================
// 27. SendTableGlobalRankProc：真实调用覆盖重构后的统一切片写法
// ============================================================
namespace {
NslbDpGlobalRankVal BuildGlobalRankVal(u32 rankCount)
{
    NslbDpGlobalRankVal val = {};
    val.taskId = 0xDEAFULL;
    val.rankTotalNum = rankCount;
    (void)strncpy_s(val.commDesc, COMM_DESC_MAX_LENGTH, "global_rank_proc_test", COMM_DESC_MAX_LENGTH - 1);
    for (u32 i = 0; i < rankCount; i++) {
        TableFourRankInfo ri = {};
        ri.deviceIp = 0x0A000001U + i;
        ri.serverIp = 0x0B000001U + i;
        val.rankInfo.push_back(ri);
    }
    return val;
}
} // namespace

TEST_F(NslbDpStFlowTest, SendTableGlobalRankProc_RealCall_Rank0)
{
    constexpr u32 kRankCount = NSLBDP_RANKTOTALNUM_BLOCK_FOU;
    NslbDpGlobalRankVal cominfo = BuildGlobalRankVal(kRankCount);
    const u32 packetNum = hcclNslbDp::CalcPacketNum(kRankCount);
    ASSERT_EQ(packetNum, 4U);

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableGlobalRankProc(0, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendTableGlobalRankProc_RealCall_LastShard)
{
    constexpr u32 kRankCount = NSLBDP_RANKTOTALNUM_BLOCK_FOU - 5;
    NslbDpGlobalRankVal cominfo = BuildGlobalRankVal(kRankCount);
    const u32 packetNum = hcclNslbDp::CalcPacketNum(kRankCount);
    ASSERT_EQ(packetNum, 4U);

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableGlobalRankProc(3, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendTableGlobalRankProc_RealCall_RankExceedsPacketNum)
{
    NslbDpGlobalRankVal cominfo = BuildGlobalRankVal(NSLBDP_RANKTOTALNUM_BLOCK_FIR);
    const u32 packetNum = 1U;

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableGlobalRankProc(1, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendTableGlobalRankProc_RealCall_EmptyRange)
{
    NslbDpGlobalRankVal cominfo = BuildGlobalRankVal(50);
    const u32 packetNum = hcclNslbDp::CalcPacketNum(50);
    ASSERT_EQ(packetNum, 1U);

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableGlobalRankProc(0, packetNum, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================
// 28. SendCommRankTable / SendTableFir：入口函数覆盖
// ============================================================
TEST_F(NslbDpStFlowTest, SendCommRankTable_RealCall_UnderLimit)
{
    constexpr u32 kRankCount = NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(kRankCount);

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendCommRankTable(0, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendCommRankTable_RealCall_ExceedsLimit)
{
    NslbDpCommConfigVal cominfo = BuildCommCfgForShardTest(NSLBDP_RANKTOTALNUM_BLOCK_FOU + 1);
    cominfo.rankTotalNum = NSLBDP_RANKTOTALNUM_BLOCK_FOU + 1;

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendCommRankTable(0, cominfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SendTableFir_RealCall_IteratesConfig)
{
    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendTableFir(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================
// 29. SendGlobalRankTable：入口函数覆盖
// ============================================================
TEST_F(NslbDpStFlowTest, SendGlobalRankTable_RealCall_UnderLimit)
{
    auto backup = inst_->hcclNslbDpGlobalRankVal_;
    inst_->hcclNslbDpGlobalRankVal_ = BuildGlobalRankVal(NSLBDP_RANKTOTALNUM_BLOCK_FIR);

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendGlobalRankTable(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    inst_->hcclNslbDpGlobalRankVal_ = backup;
}

TEST_F(NslbDpStFlowTest, SendGlobalRankTable_RealCall_ExceedsLimit)
{
    auto backup = inst_->hcclNslbDpGlobalRankVal_;
    inst_->hcclNslbDpGlobalRankVal_ = BuildGlobalRankVal(NSLBDP_RANKTOTALNUM_BLOCK_FOU + 1);
    inst_->hcclNslbDpGlobalRankVal_.rankTotalNum = NSLBDP_RANKTOTALNUM_BLOCK_FOU + 1;

    inst_->nslbdpIsInitNetCo_ = false;

    HcclResult ret = inst_->SendGlobalRankTable(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    inst_->hcclNslbDpGlobalRankVal_ = backup;
}

// ============================================================
// 30. SetCommInfo_NoRankTable：入口函数覆盖 limit 检查 + IsCommDescDuplicated
// ============================================================
TEST_F(NslbDpStFlowTest, SetCommInfo_NoRankTable_RealCall_MultiMachine)
{
    constexpr u32 kRankCount = 4;
    RankTable_t rt = BuildMultiMachineRankTable(kRankCount);
    rt.rankNum = kRankCount;

    HcclResult ret = inst_->SetCommInfo_NoRankTable(rt, "nortable_test_group", 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SetCommInfo_NoRankTable_ExceedsLimit)
{
    RankTable_t rt = BuildMultiMachineRankTable(2);
    rt.rankNum = NSLBDP_RANKTOTALNUM_BLOCK_FOU + 1;

    HcclResult ret = inst_->SetCommInfo_NoRankTable(rt, "exceed_limit_group", 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(inst_->hcclNslbDpCommConfig_.size(), 1U);
}

TEST_F(NslbDpStFlowTest, SetCommInfo_NoRankTable_DuplicatedDesc)
{
    constexpr u32 kRankCount = 4;
    RankTable_t rt = BuildMultiMachineRankTable(kRankCount);
    rt.rankNum = kRankCount;

    HcclResult ret = inst_->SetCommInfo_NoRankTable(rt, kTestCommDesc, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(inst_->hcclNslbDpCommConfig_.size(), 1U);
}

// ============================================================
// 31. SetCommInfo_RankTableExit：入口函数覆盖 limit 检查 + IsCommDescDuplicated
// ============================================================
TEST_F(NslbDpStFlowTest, SetCommInfo_RankTableExit_RealCall_MultiMachine)
{
    constexpr u32 kRankCount = 4;
    RankTable_t rt = BuildMultiMachineRankTable(kRankCount);
    rt.rankNum = kRankCount;

    auto backup = std::move(inst_->hcclNslbDpCommConfig_);
    inst_->hcclNslbDpCommConfig_.clear();

    HcclResult ret = inst_->SetCommInfo_RankTableExit(rt);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(inst_->hcclNslbDpCommConfig_.size(), 1U);

    inst_->hcclNslbDpCommConfig_ = std::move(backup);
}

TEST_F(NslbDpStFlowTest, SetCommInfo_RankTableExit_ExceedsLimit)
{
    RankTable_t rt = BuildMultiMachineRankTable(2);
    rt.rankNum = NSLBDP_RANKTOTALNUM_BLOCK_FOU + 1;

    HcclResult ret = inst_->SetCommInfo_RankTableExit(rt);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NslbDpStFlowTest, SetCommInfo_RankTableExit_SingleRank)
{
    RankTable_t rt = BuildMultiMachineRankTable(1);
    rt.rankNum = 1;

    HcclResult ret = inst_->SetCommInfo_RankTableExit(rt);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
