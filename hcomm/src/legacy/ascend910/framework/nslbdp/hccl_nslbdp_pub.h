/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NSLBDP_PUB_H
#define HCCL_NSLBDP_PUB_H

#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>

#include "hccl/base.h"
#include "hccl_common.h"
#include "hccl_comm_pub.h"
#include "mem_device_pub.h"
#include "topoinfo_struct.h"
#include "sal_pub.h"
#include "comm.h"
#include "topoinfo_exchange_agent.h"
#include "transport_heterog_def.h"
#include "../common/src/topo/topoinfo_detect.h"
#include "coll_alg_param.h"

namespace hccl {

const u32 COMM_DESC_MAX_LENGTH = 128; // group name max length

const u32 COMM_MAX_GLOABLE_ROOTRANK = 256; // group name max length

// 通信域信息表RankInfo
struct NslbDpRankInfo {
    u32 deviceIp;
    u32 serverIp;
    u16 podId;
    u16 rev;
};

struct NslbDeviceIp {
    u32 deviceIp;
};

// 表4 RankInfo
struct TableFourRankInfo {
    u32 deviceIp;
    u32 serverIp;
};

// 分表1-通信域信息表val
struct NslbDpCommConfigVal {
    u64 taskId;
    char commDesc[COMM_DESC_MAX_LENGTH];
    u64 commInitTime;
    u16 rankTotalNum;
    u8 commMd5Sum[16];
    std::vector<NslbDpRankInfo> rankInfo;
};

// 分表1-通信域信息表
struct NslbDpCommConfigInfo {
    u64 taskId;
    char commDesc[COMM_DESC_MAX_LENGTH];
    u64 commInitTime;
    u16 packetId;
    u16 rev;
    u16 packetNum;
    u16 revSecond;
    NslbDeviceIp sendRankInfo[4];
    u8 commMd5Sum[16];
    u16 rankTotalNum;
    u16 rankNum;
    std::vector<NslbDpRankInfo> rankInfo;
};

// 分表2-算子信息表
struct NslbDpOperatorInfo {
    u64 taskId; // 配置结构体大小
    char commDesc[COMM_DESC_MAX_LENGTH];
    u64 commInitTime;
    u8 oper;
    u8 algorithm;
    u16 rootRank;
    u64 trafficCnt;
    u16 l4SPortId;
    u16 maskLen;
    u32 sedFlag;
};

/*
// 算法信息表AdjInfo
struct NslbDpAdjInfo {
    u16 dstLocalRankId;
    u8 phaseId;
    u8 rev;
};

// 算法信息表AdjInfo
struct AdjInfo {
    u16 dstRankNum;
    u16 rev;
    std::vector<NslbDpAdjInfo> nsAdjInfo;
};
*/

// 分表3-算法邻接表
struct NslbDpAlgorithmTlv {
    u64 taskId;
    char commDesc[COMM_DESC_MAX_LENGTH];
    u8 commMd5Sum[16];
    u16 srcLocalRankId;
    u8 oper;
    u8 algorithm;
    u16 rootRank;
    u16 rev;
    u16 dstRankNum;
    u16 revsecond;
    std::vector<NslbDpAdjInfo> AdjInfo;
};

struct NslbDpAlgorithmInfo {
    u64 taskId;
    char commDesc[COMM_DESC_MAX_LENGTH];
    u8 commMd5Sum[16];
    u16 srcLocalRankId;
    u8 oper;
    u8 algorithm;
    u16 rootRank;
    u16 dstRankNum;
    std::vector<NslbDpAdjInfo> AdjInfo;
    u32 sedFlag;
};

// 分表4-全局Rank表val
struct NslbDpGlobalRankVal {
    u64 taskId;
    char commDesc[COMM_DESC_MAX_LENGTH];
    u64 commInitTime;
    u32 rankTotalNum;
    u8 commMd5Sum[16];
    std::vector<TableFourRankInfo> rankInfo;
};

// 分表4-全局Rank表
struct NslbDpGlobalRankInfo {
    u64 taskId;
    char commDesc[COMM_DESC_MAX_LENGTH];
    u64 commInitTime;
    u16 packetId;
    u16 rev;
    u16 packetNum;
    u16 rev2;
    NslbDeviceIp sendRankInfo[COMM_MAX_GLOABLE_ROOTRANK];
    u8 commMd5Sum[16];
    u32 rankTotalNum;
    u16 rankNum;
    u16 rev3;
    std::vector<TableFourRankInfo> rankInfo;
};

// 分表5-全局Rank分布式信息val
struct NslbDpGlobalDisRankVal {
    u64 taskId;
    u32 npuIp;
    u32 serverIp;
    u32 nodeId;
    u8 localRankNum;
    u8 rev[3];
    u32 rankTotalNum;
};

struct NslbDpGlobalCommInfo {
    u64 taskId;
    u32 nodeId;
    u8 localRankNum;
    u32 rankTotalNum;
};

struct NslbDpRankId {
    u16 rankID;
};

// 分表6-Root Rank分布式信息
struct NslbDpRootRank {
    u64 taskId;
    char commDesc[COMM_DESC_MAX_LENGTH];
    u64 commInitTime;
    u8 oper;
    u8 algorithm;
    u16 revfir;
    u16 rootRankNum;
    u16 revsec;
    std::vector<NslbDpRankId> rankId;
};

constexpr unsigned int TLV_SEND_TYPE_NETCO_INIT = 0;
constexpr unsigned int TLV_SEND_TYPE_NETCO_DEINIT = 1;
constexpr unsigned int TLV_SEND_TYPE_TBL_COMM_INFO = 2;
constexpr unsigned int TLV_SEND_TYPE_TBL_OPER = 3;
constexpr unsigned int TLV_SEND_TYPE_TBL_ADJ = 4;
constexpr unsigned int TLV_SEND_TYPE_TBL_RANK = 5;
constexpr unsigned int TLV_SEND_TYPE_TBL_RANK_DIST = 6;
constexpr unsigned int TLV_SEND_TYPE_TBL_ROOT_RANK = 7;

// typedef void *NslbdpCommHandle;

struct nslb_inithccp_info {
    s32 version;
    u32 phyId;
    u32 nic_posion;
    u32 reserved[16U];
};

} // namespace hccl

#endif /* HCCL_NSLB_DP_PUB_H */
