/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NSLBDP_H
#define HCCL_NSLBDP_H

#include <vector>
#include <memory>
#include <map>
#include <mutex>

#include "hccl/base.h"
#include "hccl_common.h"
#include "hccl_comm_pub.h"
#include "topoinfo_struct.h"
#include "sal_pub.h"
#include "comm.h"
#include "coll_alg_param.h"
#include "hccl_nslbdp_pub.h"

namespace hccl {

constexpr unsigned int NSLBDP_ILLEGAL_TLVBUFFERSIZE = 0;
constexpr unsigned int NSLBDP_ILLEGAL_MSGLENGTH = 0;
constexpr unsigned int NSLBDP_TYPE_TBL_COMM_INFO = 1001;
constexpr unsigned int NSLBDP_TYPE_TBL_OPER = 1002;
constexpr unsigned int NSLBDP_TYPE_TBL_ADJ = 1003;
constexpr unsigned int NSLBDP_TYPE_TBL_RANK = 1004;
constexpr unsigned int NSLBDP_TYPE_TBL_RANK_DIST = 1005;
constexpr unsigned int NSLBDP_TYPE_TBL_ROOT_RANK = 1006;
constexpr unsigned int NSLBDP_TYPE_INIT_NETCO = 9001;
constexpr unsigned int NSLBDP_TYPE_DEINIT_NETCO = 9002;

constexpr u32 NSLBDP_RANKTOTALNUM_BLOCK_FIR = 1024;
constexpr u32 NSLBDP_RANKTOTALNUM_BLOCK_SEC = 2 * 1024;
constexpr u32 NSLBDP_RANKTOTALNUM_BLOCK_THR = 3 * 1024;
constexpr u32 NSLBDP_RANKTOTALNUM_BLOCK_FOU = 4 * 1024;

constexpr u32 NSLBDP_HIGH_8BIT = 8;

constexpr u16 NSLBDP_COMMINTERVAL_FLAG = 128;
constexpr u16 NSLBDP_COMMINTERVAL_FLAGSEC = 256;
constexpr u16 NSLBDP_COMMINTERVAL_FLAGTHR = 512;
constexpr u16 NSLBDP_COMMINTERVAL_FLAGFOU = 1024;
constexpr u16 NSLBDP_COMMINTERVAL_FLAGFIV = 2048;
constexpr u16 NSLBDP_COMMINTERVAL_FLAGSIX = 4096;

constexpr u32 NSLBDP_PKTNUM_FIR = 1;
constexpr u32 NSLBDP_PKTNUM_SEC = 2;
constexpr u32 NSLBDP_PKTNUM_THR = 3;
constexpr u32 NSLBDP_PKTNUM_FOU = 4;

constexpr u32 NSLBDP_TASKID_OFFSET = 32;

constexpr u32 NSLBDP_INITIME_MILLISENDS = 1000;

constexpr u32 NSLBDP_PAIRWISE = 5;

constexpr u32 NSLBDP_BEGINFOURBIT = 2;

constexpr u32 NSLBDP_SPLIT_SIZE = 3;

constexpr u16 NSLBDP_PRIVATE_PORT = 3;
constexpr u16 NSLBDP_RANGE_ID = 14;
constexpr u16 NSLBDP_COMMON_RANGE = 11;
constexpr u16 NSLBDP_ALGO_RANGE = 4;

constexpr u32 NSLBDP_TRAFFICCONUT = 60;

enum class NslbDpAlgType {
    NSLBDP_WHOLE_RING = 0, // 单层拓扑, 所有level均为Whole ring时，组成一个大环
    NSLBDP_HD,             // HDR
    NSLBDP_RING,           // Ring
    NSLBDP_PIPELINE,       // Pipeline
    NSLBDP_STAR,
    NSLBDP_NHR,       // NHR
    NSLBDP_NHR_V1,    // NHR_V1
    NSLBDP_NB,        // NB
    NSLBDP_AHC,       // AHC
    NSLBDP_AHC_BROKE, // AHC_BROKE
    NSLBDP_RESERVED
};

constexpr u8 NSLBDP_CMD_INVALID = 0;
constexpr u8 NSLBDP_CMD_BROADCAST = 1;
constexpr u8 NSLBDP_CMD_ALLREDUCE = 2;
constexpr u8 NSLBDP_CMD_REDUCE = 3;
constexpr u8 NSLBDP_CMD_SEND = 4;
constexpr u8 NSLBDP_CMD_RECEIVE = 5;
constexpr u8 NSLBDP_CMD_ALLGATHER = 6;
constexpr u8 NSLBDP_CMD_REDUCE_SCATTER = 7;
constexpr u8 NSLBDP_CMD_ALLTOALLV = 8;
constexpr u8 NSLBDP_CMD_ALLTOALLVC = 9;
constexpr u8 NSLBDP_CMD_ALLTOALL = 10;
constexpr u8 NSLBDP_CMD_GATHER = 11;
constexpr u8 NSLBDP_CMD_SCATTER = 12;
constexpr u8 NSLBDP_CMD_BATCH_SEND_RECV = 13;
constexpr u8 NSLBDP_CMD_BATCH_PUT = 14;
constexpr u8 NSLBDP_CMD_BATCH_GET = 15;
constexpr u8 NSLBDP_CMD_ALL = 16;

constexpr u8 NSLB_ALGO_TYPE_DEFAULT = 0;
constexpr u8 NSLB_ALGO_TYPE_RING = 1;
constexpr u8 NSLB_ALGO_TYPE_PIPELINE = 2;
constexpr u8 NSLB_ALGO_TYPE_FULLMESH = 3;
constexpr u8 NSLB_ALGO_TYPE_HDR = 4;
constexpr u8 NSLB_ALGO_TYPE_PAIRWISE = 5;
constexpr u8 NSLB_ALGO_TYPE_NHR = 6;
constexpr u8 NSLB_ALGO_TYPE_NHR_V1 = 7;
constexpr u8 NSLB_ALGO_TYPE_NB = 8;
constexpr u8 NSLB_ALGO_TYPE_NULL = 9;
constexpr u8 NSLB_ALGO_TYPE_NA = 10;
constexpr u8 NSLB_ALGO_TYPE_FAST_DOUBLE_RING = 11;
constexpr u8 NSLB_ALGO_TYPE_AHC = 12;

constexpr u16 NSLB_COMM_INTERVAL_FLAG_BEGIN = 0;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_FIR = 1;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_SEC = 2;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_THR = 3;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_FOR = 4;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_FIV = 5;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_SIX = 6;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_SEV = 7;
constexpr u16 NSLB_COMM_INTERVAL_FLAG_MAX = 8;

constexpr unsigned int MODULE_TYPE_NSLB = 0;
constexpr unsigned int MODULE_TYPE_MAX = 1;

constexpr u32 NSLBDP_UNDERDCORES_COUNT = 3;

using nslb_msg = struct nslb_msg {
    unsigned int type;
    unsigned int length;
    std::string data;

    nslb_msg() : type(INVALID_UINT), length(0), data("") {}
};

class hcclNslbDp {
public:
    static hcclNslbDp& GetInstance();
    NslbDpCommConfigVal GetNslbDpCommConfig();
    HcclResult HcclSetGlobalRankTotalNum(u32 nRanks);

    bool GetInitNetCoFlag();
    HcclResult ClearInitNetCoFlag();
    void InitCmmDesc(std::string& identifier_nslb);
    std::string GetCmmDesc();
    void SetDeviceType();
    void SetGlobalCommTaskId(u64 taskId);
    void SetGlobalCommNodeId(u32 nodeId);
    void SetGlobalCommLocalRankNum(u32 localRankNum);
    void SetGlobalCommRankTotalNum(u32 rankTotalNum);
    bool GetDeviceType();
    u64 GetGlobalCommTaskId();
    u32 GetGlobalCommNodeId();
    u32 GetTlvInitBufferSize();
    u8 GetGlobalCommLocalRankNum();
    u8 GetNslbOpType(HcclCMDType opType);
    u8 GetNslbLevel1AlgType(AlgTypeLevel1 algValue);
    u8 GetNslbLevel2AlgType(AlgTypeLevel2 algValue);
    u32 GetGlobalCommRankTotalNum();
    u32 Getl4SPortId();
    u64 GetNslbDpFirstFourBit(u8 opType, u8 algType);
    bool CheckAlgoConsistency(HcclCMDType opType, std::string& algName);
    void SplitString(const std::string& identifier, std::vector<std::string>& splitInfo, const std::string& frag);
    void SetGlobalDisRankTable(const HcclBasicRankInfo& rankTable);
    HcclResult SetCommInfo_NoRankTable(const RankTable_t rankTable, std::string identifier, u32 subCommRankId);
    HcclResult SetCommInfo_RankTableExit(RankTable_t rankTable);
    HcclResult SetGlobalRank_RankTableExit(const RankTable_t rankTable);
    HcclResult GenerateOpAndAdjTable(
        HcclCMDType opType, u32 rootRank, u32 srcLocalRankId, u8 algType, std::string identifier, u64 count,
        u32 rankSize);
    HcclResult GetAlgAdjacencyTable(
        HcclCMDType opType, u32 srcLocalRankId, u32 rootRank, u8 algType, std::string identifier, AdjInfo nslbAdjInfo);
    HcclResult GetNslbDpl4SPortId(u32 rankSize, u8 algType, u16* l4SPortId);
    HcclResult SendCommRankTable(uint32_t rank, NslbDpCommConfigVal globalCommInfo);
    bool CheckMultiMachine(const RankTable_t rankTable);
    bool CheckSupportOptype(HcclCMDType opType);
    bool CheckCommDescExit(NslbDpOperatorInfo& OperatorInfo);
    bool CheckSameOperatorVal(size_t operSize, NslbDpOperatorInfo& OperatorInfo, u32 rootRank);
    bool IsCommDescDuplicated(const char* commDesc, u64 taskId) const;
    void FillRankInfoFromRankTable(NslbDpCommConfigVal& globalCommInfo, const hccl::RankTable_t& rankTable);
    void SetGlobalCommRankTable_RootInfo(
        const RankTable_t& rankTable, const HcclBasicRankInfo& localRankInfo, const std::vector<RankInfo>& rankLists,
        const std::string& identifier, u32 nRanks, u32 rank);
    void GetGlobalRankTable(const RankTable_t* rankTable, u32 nRanks, HcclUs startut);
    void fullCommonGlobalRankInfo(NslbDpGlobalRankInfo& tab_f, NslbDpGlobalRankVal& cominfo);
    void fullCommConfigInfo(NslbDpCommConfigInfo& tab_f, NslbDpCommConfigVal cominfo, u32 packetNum);
    void fullcommDescInitTime(std::string identifier, NslbDpOperatorInfo& OperatorInfo);
    HcclResult SetNslbDpRootRank(HcclCMDType opType, u32 rootRank, std::string identifier, u8 algType);
    HcclResult SendRankTable(NslbDpCommConfigInfo tab_f);
    std::vector<uint8_t> serializeTLV_TableFir(NslbDpCommConfigInfo cominfo);
    HcclResult SendTableProc(u32 rank, u32 packetNum, NslbDpCommConfigVal cominfo);
    HcclResult SendTableFir(uint32_t rank);
    // 计算总分片数：ceil(rankTotalNum / BLOCK_FIR)，区间内对应 1/2/3/4 分片
    static u32 CalcPacketNum(u32 rankTotalNum)
    {
        if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_FIR) {
            return NSLBDP_PKTNUM_FIR;
        } else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_SEC) {
            return NSLBDP_PKTNUM_SEC;
        } else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_THR) {
            return NSLBDP_PKTNUM_THR;
        } else {
            return NSLBDP_PKTNUM_FOU;
        }
    }
    HcclResult SetH2DTlvInitInfo(u32 buffer_size, void* tlv_handle);
    static u32 ipToUint32(const std::string& ipAddress);
    /* 将serverId解析为u32：先剥掉 _<LogicSuperPodId> 后缀再转IP，保证各处口径一致 */
    static u32 ServerIdToIp(const std::string& serverId);
    HcclResult SendOpAndAdjTable();
    HcclResult SendRankTableOpAndAdj(NslbDpOperatorInfo& tab_f);
    std::vector<uint8_t> serializeTLV_TableOpAndAdj(NslbDpOperatorInfo& info);
    HcclResult SendAlgorithmInfoTable();
    HcclResult SendRankTableAlgorithmInfo(NslbDpAlgorithmTlv& tab_f);
    std::vector<uint8_t> serializeTLV_TableAlgorithmInfo(NslbDpAlgorithmTlv& info);
    HcclResult SendGlobalRankTable(uint32_t rank);
    HcclResult SendTableGlobalRankProc(uint32_t rank, uint32_t packetNum, NslbDpGlobalRankVal& cominfo);
    std::vector<uint8_t> serializeTLV_TableGlobalRankInfo(NslbDpGlobalRankInfo& info);
    HcclResult SendRankTableGlobalRank(NslbDpGlobalRankInfo& tab_f);
    HcclResult SendGlobalDisRankTable();
    HcclResult SendRankTableGlobalDisRankVal(NslbDpGlobalDisRankVal& tab_f);
    std::vector<uint8_t> serializeTLV_TableGlobalDisRankVal(NslbDpGlobalDisRankVal& info);
    HcclResult SendRootRankTable();
    HcclResult SendRankTableRootRank(NslbDpRootRank& tab_f);
    std::vector<uint8_t> serializeTLV_TableRootRank(NslbDpRootRank& config);
    HcclResult InitNetCo();
    void DeinitNetCo();
    bool check910_93_ = false;
    std::atomic<bool> nslbdpIsInitNetCo_ = {false};
    void* nslbdp_handle_;
    unsigned int nslbdp_buffsize_;
    std::string nslbdp_identifier_;
    u32 hcclNslbDpL4SPortId_;
    // 上层调用数据存储关键字唯一标识的taskid等信息
    NslbDpGlobalCommInfo hcclNslbDpGlobalCommInfo_;
    // 分表1-基础数据. 承载通信与信息
    std::vector<NslbDpCommConfigVal> hcclNslbDpCommConfig_;
    // 分表2-基础数据，承载执行的算子算法信息
    std::vector<NslbDpOperatorInfo> hcclNslbDpOperatorVal_;
    // 分表3-基础数据，承载执行的算子算法的邻接信息
    std::vector<NslbDpAlgorithmInfo> hcclNslbDpAlgorithmInfo_;
    // 分表4-基础数据，在非ranktble场景下创建通信与场景承载全局通信域信息
    NslbDpGlobalRankVal hcclNslbDpGlobalRankVal_;
    // 分表5-基础数据，在非ranktble场景下创建通信与场景域场景下的分布式rank表
    NslbDpGlobalDisRankVal hcclNslbDpGlobalDisRankVal_;
    // 分表6-基础数据，非对称算子 statter， reduce，bcast 算子场景下会有rootrabnk表
    NslbDpRootRank hcclNslbDpRootRankVal_;

private:
    hcclNslbDp();
    ~hcclNslbDp();
    bool CheckAhcCommInfo(NslbDpCommConfigVal comInfo);
    bool CheckAhcSupport(u8 algType, std::string identifier);
    bool InitAlgInfoCommDesc(NslbDpAlgorithmInfo& algorithmInfo, const std::string& identifier);
    // 按identifier取表1MD5，再取同MD5中commInitTime最小的commDesc；失败返回false
    bool FindMinInitTimeCommDesc(const std::string& identifier, std::string& commDesc, u64& minInitTime);
    // 命中"存在不同commDesc但MD5相同的已有条目"时返回true，用于跳过重复下发表1
    bool ShouldSkipSendByMd5(const NslbDpCommConfigVal& info);
    bool FillAlgInfoCommMd5(NslbDpAlgorithmInfo& algorithmInfo);
    void FillAlgInfoBaseFields(
        NslbDpAlgorithmInfo& algorithmInfo, HcclCMDType opType, u32 srcLocalRankId, u32 rootRank, u8 algType);
    bool TryFillA3SimulatedAdjInfo(const std::string& identifier, u32 srcLocalRankId, AdjInfo& nslbAdjInfo);
    bool IsAlgAdjacencyDuplicated(const NslbDpAlgorithmInfo& algorithmInfo);
    bool FillAlgInfoAdjInfo(NslbDpAlgorithmInfo& algorithmInfo, const AdjInfo& nslbAdjInfo, u32 srcLocalRankId);
};

} // namespace hccl

#endif /* HCCL_NSLB_DP_PUB_H */
