/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <hccl/hccl_types.h>
#include "topoinfo_struct.h"
#include "adapter_hccp_common.h"
#include "hccl_nslb_md5.h"
#include "../common/src/h2d_tlv/hccl_h2dtlv.h"
#include "hccl_nslbdp.h"

namespace hccl {

const u32 COMM_SLICING_LENGTH = 1024; // group name max length

hcclNslbDp& hcclNslbDp::GetInstance()
{
    static hcclNslbDp pNslbDp;
    return pNslbDp;
}

hcclNslbDp::hcclNslbDp() {}

hcclNslbDp::~hcclNslbDp() {}

void hcclNslbDp::InitCmmDesc(std::string& identifier_nslb)
{
    nslbdp_identifier_ = identifier_nslb;
    HCCL_INFO("[NSLB-DP] Init CmmDesc nslbdp_identifier_[%s] .", nslbdp_identifier_.c_str());
    return;
}

std::string hcclNslbDp::GetCmmDesc()
{
    HCCL_INFO("[NSLB-DP] Get CmmDesc nslbdp_identifier_[%s] .", nslbdp_identifier_.c_str());
    return nslbdp_identifier_;
}

void hcclNslbDp::SetDeviceType()
{
    check910_93_ = true;
    return;
}

bool hcclNslbDp::GetDeviceType()
{
    bool is91093 = check910_93_;
    return is91093;
}

HcclResult hcclNslbDp::SetH2DTlvInitInfo(u32 buffer_size, void* tlv_handle)
{
    HCCL_DEBUG("[NSLB-DP] try to set tlvinit info.");
    nslbdp_buffsize_ = buffer_size;
    nslbdp_handle_ = tlv_handle;
    nslbdpIsInitNetCo_ = true;
    HCCL_INFO("[NSLB-DP] set tlvinit info buffer_size:[%u].", buffer_size);
    return HCCL_SUCCESS;
}

bool hcclNslbDp::GetInitNetCoFlag() { return nslbdpIsInitNetCo_; }

HcclResult hcclNslbDp::ClearInitNetCoFlag()
{
    nslbdpIsInitNetCo_ = false;
    return HCCL_SUCCESS;
}

u32 hcclNslbDp::GetTlvInitBufferSize()
{
    u32 nslbdpBuffsize = nslbdp_buffsize_;
    return nslbdpBuffsize;
}

HcclResult hcclNslbDp::HcclSetGlobalRankTotalNum(u32 nRanks)
{
    // 填充表4
    hcclNslbDpGlobalRankVal_.rankTotalNum = nRanks;
    return HCCL_SUCCESS;
}

void hcclNslbDp::SetGlobalCommTaskId(u64 taskId)
{
    hcclNslbDpGlobalCommInfo_.taskId = taskId;
    return;
}

void hcclNslbDp::SetGlobalCommNodeId(u32 nodeId)
{
    hcclNslbDpGlobalCommInfo_.nodeId = nodeId;
    return;
}

void hcclNslbDp::SetGlobalCommLocalRankNum(u32 localRankNum)
{
    hcclNslbDpGlobalCommInfo_.localRankNum = localRankNum;
    return;
}

void hcclNslbDp::SetGlobalCommRankTotalNum(u32 rankTotalNum)
{
    hcclNslbDpGlobalCommInfo_.rankTotalNum = rankTotalNum;
    return;
}

u64 hcclNslbDp::GetGlobalCommTaskId() { return hcclNslbDpGlobalCommInfo_.taskId; }

u32 hcclNslbDp::GetGlobalCommNodeId() { return hcclNslbDpGlobalCommInfo_.nodeId; }

u8 hcclNslbDp::GetGlobalCommLocalRankNum() { return hcclNslbDpGlobalCommInfo_.localRankNum; }

u32 hcclNslbDp::GetGlobalCommRankTotalNum() { return hcclNslbDpGlobalCommInfo_.rankTotalNum; }

/* 获取 l4SPortId */
u32 hcclNslbDp::Getl4SPortId()
{
    u32 NslbDpL4SPortId = hcclNslbDpL4SPortId_;
    return NslbDpL4SPortId;
}

/* 切分identifier字段 */
void hcclNslbDp::SplitString(
    const std::string& identifier, std::vector<std::string>& splitInfo, const std::string& frag)
{
    std::string::size_type pos2 = identifier.find(frag);
    std::string::size_type pos1 = 0;
    while (std::string::npos != pos2) {
        splitInfo.push_back(identifier.substr(pos1, pos2 - pos1));
        pos1 = pos2 + frag.size();
        pos2 = identifier.find(frag, pos1);
    }
    if (pos1 != identifier.length()) {
        splitInfo.push_back(identifier.substr(pos1));
    }
    return;
}

/* 将IP转换成U32值 */
u32 hcclNslbDp::ipToUint32(const std::string& ipAddress)
{
    struct sockaddr_in sa {};
    if (inet_pton(AF_INET, ipAddress.c_str(), &(sa.sin_addr)) != 1) {
        HCCL_INFO("[NSLB-DP] ipToUint32 parse ip[%s] fail, return 0.", ipAddress.c_str());
        return 0;
    }
    return ntohl(sa.sin_addr.s_addr); // Convert to host byte order
}

/* 解析serverId对应的u32值：先剥掉 _<LogicSuperPodId> 后缀再转IP，避免垃圾值流入MD5 */
u32 hcclNslbDp::ServerIdToIp(const std::string& serverId)
{
    std::string ip = serverId;
    size_t underscorePos = ip.find('_');
    if (underscorePos != std::string::npos) {
        ip = ip.substr(0, underscorePos);
    }
    return hcclNslbDp::ipToUint32(ip);
}

/* 执行send 流程 */
HcclResult hcclNslbDp::SendCommRankTable(uint32_t rank, NslbDpCommConfigVal globalCommInfo)
{
    HCCL_DEBUG("[NSLB-DP] entry to send TBL_COMM_INFO");
    u32 rankTotalNum = globalCommInfo.rankTotalNum;

    HCCL_INFO("[NSLB-DP] TBL_COMM_INFO rankTotalNum:[%u].", rankTotalNum);
    if (rankTotalNum > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
        return HCCL_SUCCESS;
    }
    u32 packetNum = CalcPacketNum(rankTotalNum);
    SendTableProc(rank, packetNum, globalCommInfo);

    HCCL_DEBUG("[NSLB-DP] entry send TBL_COMM_INFO end");
    return HCCL_SUCCESS;
}

bool hcclNslbDp::CheckAhcCommInfo(NslbDpCommConfigVal comInfo)
{
    std::unordered_map<uint32_t, int> podToIpCount;
    for (const auto& info : comInfo.rankInfo) {
        podToIpCount[info.podId]++;
    }
    // 获取超节点对应的ip数量
    std::vector<int> ipCounts;
    for (const auto& entry : podToIpCount) {
        ipCounts.push_back(entry.second);
    }
    if (ipCounts.empty()) {
        return true;
    }
    // 找到最小的ip数量
    uint32_t minCount = *std::min_element(ipCounts.begin(), ipCounts.end());
    for (uint32_t count : ipCounts) {
        if (count % minCount != 0) {
            return false;
        }
    }
    return true;
}

bool hcclNslbDp::CheckAhcSupport(u8 algType, std::string identifier)
{
    if (algType != NSLB_ALGO_TYPE_AHC) {
        return true;
    }
    char commDesc[COMM_DESC_MAX_LENGTH];
    s32 sRet = memset_s(commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memset_s commDesc fail");
        return true;
    }

    s32 ret = strncpy_s(commDesc, COMM_DESC_MAX_LENGTH, identifier.c_str(), identifier.size());
    if (ret != EOK) {
        HCCL_INFO("strncpy_s commDesc fail");
        return true;
    }
    commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    for (const auto& info : hcclNslbDpCommConfig_) {
        if (strcmp(commDesc, info.commDesc) == 0) {
            return CheckAhcCommInfo(info);
        }
    }
    return true;
}

/* 判断指定 commDesc + taskId 的表一是否已存在(避免重复填充, inittime 必然不同) */
bool hcclNslbDp::IsCommDescDuplicated(const char* commDesc, u64 taskId) const
{
    for (size_t i = 0; i < hcclNslbDpCommConfig_.size(); i++) {
        if (hcclNslbDpCommConfig_[i].taskId == taskId && strcmp(hcclNslbDpCommConfig_[i].commDesc, commDesc) == 0) {
            return true;
        }
    }
    return false;
}

/* 从 rankTable.rankList 构建并填充 rankInfo 列表到 globalCommInfo */
void hcclNslbDp::FillRankInfoFromRankTable(NslbDpCommConfigVal& globalCommInfo, const hccl::RankTable_t& rankTable)
{
    for (u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
        NslbDpRankInfo dpRankInfo;
        HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
        std::string deviceIp = tmpIp.GetReadableAddress();
        dpRankInfo.deviceIp = ipToUint32(deviceIp);
        HCCL_INFO("[NSLB-DP] FillRankInfoFromRankTable deviceIp:[%u] success.", dpRankInfo.deviceIp);

        std::string serverIp = rankTable.rankList[rankIndex].serverId;
        dpRankInfo.serverIp = ServerIdToIp(serverIp);
        HCCL_INFO("[NSLB-DP] FillRankInfoFromRankTable serverIp:[%u] success.", dpRankInfo.serverIp);

        if (rankTable.rankList[rankIndex].superPodIdx == INVALID_UINT) {
            dpRankInfo.podId = 0;
        } else {
            dpRankInfo.podId = rankTable.rankList[rankIndex].superPodIdx;
        }
        dpRankInfo.rev = 0;
        globalCommInfo.rankInfo.push_back(dpRankInfo);
    }
}

/* 读配置文件的场景与正常创建通信域场景下填充通信域信息表（表一） */
void hcclNslbDp::SetGlobalCommRankTable_RootInfo(
    const RankTable_t& rankTable, const HcclBasicRankInfo& localRankInfo, const std::vector<RankInfo>& rankLists,
    const std::string& identifier, u32 nRanks, u32 rank)
{
    HCCL_INFO(
        "[NSLB-DP] Try to collect NSLBDP_TYPE_TBL_COMM_INFO commDesc[%s] - size = [%u].", identifier.c_str(),
        rankTable.rankList.size());
    u64 checkTaskId = GetGlobalCommTaskId();
    if (checkTaskId == 0 || nRanks == 1) {
        return;
    }
    if (nRanks > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
        HCCL_INFO("[NSLB-DP] nRanks[%u] exceeds limit[%u], skip TBL_COMM_INFO.", nRanks, NSLBDP_RANKTOTALNUM_BLOCK_FOU);
        return;
    }

    // 判断是否跨机
    if (CheckMultiMachine(rankTable) == false) {
        HCCL_INFO("[NSLB-DP] CheckMultiMachine is false");
        return;
    }

    NslbDpCommConfigVal globalCommInfo{};
    u32 sRet = memset_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(globalCommInfo.commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memset_s commDesc fail sRet[%u]", sRet);
        return;
    }
    /* 获取通信域唯一标识 */
    sRet = strncpy_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH, identifier.c_str(), identifier.size());
    if (sRet != EOK) {
        HCCL_ERROR("strncpy_s commDesc fail sRet[%u]", sRet);
        return;
    }
    globalCommInfo.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

    if (IsCommDescDuplicated(globalCommInfo.commDesc, checkTaskId)) {
        HCCL_INFO(
            "[NSLB-DP] commDesc[%s] taskId[%llu] already exists, skip TBL_COMM_INFO.", identifier.c_str(), checkTaskId);
        return;
    }

    u64 utime
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
              .count();
    globalCommInfo.commInitTime = utime;
    globalCommInfo.taskId = checkTaskId;
    globalCommInfo.rankTotalNum = nRanks;

    for (u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
        NslbDpRankInfo dpRankInfo;
        HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
        std::string deviceIp = tmpIp.GetReadableAddress();
        dpRankInfo.deviceIp = ipToUint32(deviceIp);
        HCCL_INFO("[NSLB-DP] SetGlobalCommRankTable_RootInfo deviceIp:[%u] success.", dpRankInfo.deviceIp);

        std::string serverIp = rankTable.rankList[rankIndex].serverId;
        dpRankInfo.serverIp = ServerIdToIp(serverIp);

        HCCL_INFO("[NSLB-DP] SetGlobalCommRankTable_RootInfo serverIp:[%s] success.", serverIp.c_str());
        if (rankLists.size() < rankIndex) {
            return;
        }
        if (rankLists[rankIndex].superPodIdx == INVALID_UINT) {
            dpRankInfo.podId = 0;
        } else {
            dpRankInfo.podId = rankLists[rankIndex].superPodIdx;
        }
        dpRankInfo.rev = 0;
        globalCommInfo.rankInfo.push_back(dpRankInfo);
    }
    NSLBMD5::calculateRankInfoMd5(globalCommInfo.rankInfo, globalCommInfo.commMd5Sum);
    std::string nslbdpmd5 = NSLBMD5::md5ToString(globalCommInfo.commMd5Sum);
    HCCL_DEBUG("[NSLB-DP] NSLBDP-MD5 nslbdpmd5:[%s] success.", nslbdpmd5.c_str());

    std::string npuIp = localRankInfo.deviceIP[0].GetReadableIP();
    if (ipToUint32(npuIp) != 0) {
        /* 检查是否存在不同 commDesc 但 MD5 相同的已有条目，若存在则跳过发送 */
        if (ShouldSkipSendByMd5(globalCommInfo)) {
            HCCL_INFO(
                "[NSLB-DP] rank[%u]: identifier[%s] same MD5 with diff commDesc, skip send.", rank, identifier.c_str());
        } else {
            HCCL_RUN_INFO(
                "[NSLB-DP] rank[%u]: identifier[%s] nslbdpmd5:[%s].", rank, identifier.c_str(), nslbdpmd5.c_str());
            SendCommRankTable(rank, globalCommInfo);
        }
    }
    hcclNslbDpCommConfig_.push_back(globalCommInfo);
    HCCL_INFO("[NSLB-DP] Entry SetGlobalCommRankTable_RootInfo end size = [%zu]", hcclNslbDpCommConfig_.size());
}

/* 填充表5 */
void hcclNslbDp::SetGlobalDisRankTable(const HcclBasicRankInfo& rankTable)
{
    HCCL_INFO("[NSLB-DP] Try to collect NSLBDP_TYPE_TBL_RANK_DIST");
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return;
    }
    std::string npuIp = rankTable.deviceIP[0].GetReadableIP();
    if (ipToUint32(npuIp) == 0) {
        return;
    }
    hcclNslbDpGlobalDisRankVal_.taskId = taskId;
    hcclNslbDpGlobalDisRankVal_.nodeId = GetGlobalCommNodeId();
    hcclNslbDpGlobalDisRankVal_.localRankNum = GetGlobalCommLocalRankNum();
    hcclNslbDpGlobalDisRankVal_.rankTotalNum = GetGlobalCommRankTotalNum();

    std::string serverIp = rankTable.hostIP.GetReadableIP();
    hcclNslbDpGlobalDisRankVal_.serverIp = ipToUint32(serverIp);
    HCCL_DEBUG("[NSLB-DP] SetGlobalDisRankTable serverIp:[%u].", hcclNslbDpGlobalDisRankVal_.serverIp);

    hcclNslbDpGlobalDisRankVal_.npuIp = ipToUint32(npuIp);
    HCCL_DEBUG("[NSLB-DP] SetGlobalDisRankTable npuIp:[%u].", hcclNslbDpGlobalDisRankVal_.npuIp);
}

/* check 是否是多机场景 */
bool hcclNslbDp::CheckMultiMachine(const RankTable_t rankTable)
{
    HCCL_DEBUG("[NSLB-DP] check device is multi machine");
    u32 serverIpFir = 0;
    u16 podIdFir = 0;
    bool bIsMultiMachine = false;
    if (GetDeviceType() == true) {
        u16 podIdIndex = 0;
        for (u32 index = 0; index < rankTable.rankList.size(); index++) {
            if (rankTable.rankList[index].superPodIdx == INVALID_UINT) {
                podIdIndex = 0;
            } else {
                podIdIndex = rankTable.rankList[index].superPodIdx;
            }
            if (index == 0) {
                podIdFir = podIdIndex;
                continue;
            }

            if (podIdFir != podIdIndex) {
                return true;
            }
        }
        return false;
    }
    for (u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
        std::string serverIpInfo = rankTable.rankList[rankIndex].serverId;
        u32 serverIp = ipToUint32(serverIpInfo);
        u16 podIdInfo = 0;
        if (rankTable.rankList[rankIndex].superPodIdx == INVALID_UINT) {
            podIdInfo = 0;
        } else {
            podIdInfo = rankTable.rankList[rankIndex].superPodIdx;
        }
        if (rankIndex == 0) {
            serverIpFir = serverIp;
            podIdFir = podIdInfo;
            continue;
        }

        if (serverIp != serverIpFir || podIdFir != podIdInfo) {
            bIsMultiMachine = true;
            break;
        }
    }
    return bIsMultiMachine;
}

/* 无ranktable场景， 子通信域场景表1 赋值 */
HcclResult
hcclNslbDp::SetCommInfo_NoRankTable(const hccl::RankTable_t rankTable, std::string identifier, u32 subCommRankId)
{
    HCCL_DEBUG("[NSLB-DP] Try to collect NSLBDP_TYPE_TBL_COMM_INFO for no RankTable");
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    if (rankTable.rankNum > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
        HCCL_INFO(
            "[NSLB-DP] rankNum[%u] exceeds limit[%u], skip TBL_COMM_INFO.", rankTable.rankNum,
            NSLBDP_RANKTOTALNUM_BLOCK_FOU);
        return HCCL_SUCCESS;
    }
    // 判断是否跨机 false非错误场景
    if (CheckMultiMachine(rankTable) == false) {
        HCCL_INFO("[NSLB-DP] nslb-dp CheckMultiMachine is false");
        return HCCL_SUCCESS;
    }

    NslbDpCommConfigVal globalCommInfo;
    (void)memset_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(globalCommInfo.commDesc));

    // 获取通信域唯一标识
    s32 ret = strncpy_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH, identifier.c_str(), identifier.size());
    if (ret != EOK) {
        HCCL_INFO("strncpy_s globalCommInfo.commDesc fail");
        return HCCL_E_MEMORY;
    }
    globalCommInfo.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';
    if (IsCommDescDuplicated(globalCommInfo.commDesc, taskId)) {
        HCCL_INFO(
            "[NSLB-DP] commDesc[%s] taskId[%llu] already exists, skip TBL_COMM_INFO.", identifier.c_str(), taskId);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[NSLB_DP]GetIdentifier str copy fail. return[%d]", ret), HCCL_E_INTERNAL);

    // commInitTime在有ranktable的赋值
    u64 utime
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
              .count();
    globalCommInfo.commInitTime = utime;
    u16 nRanks = rankTable.rankNum;
    globalCommInfo.taskId = taskId;
    globalCommInfo.rankTotalNum = nRanks;

    FillRankInfoFromRankTable(globalCommInfo, rankTable);
    NSLBMD5::calculateRankInfoMd5(globalCommInfo.rankInfo, globalCommInfo.commMd5Sum);
    std::string nslbdpmd5 = NSLBMD5::md5ToString(globalCommInfo.commMd5Sum);

    /* 存在不同 commDesc 但 MD5 相同的已有条目时，跳过下发表1，仅记录 */
    if (ShouldSkipSendByMd5(globalCommInfo)) {
        HCCL_INFO(
            "[NSLB-DP] Subcomm rankId[%u] identifier[%s] same MD5 with diff commDesc, skip send.", subCommRankId,
            identifier.c_str());
    } else {
        HCCL_RUN_INFO(
            "[NSLB-DP] Subcomm rankId[%u] identifier[%s] nslbdpmd5:[%s].", subCommRankId, identifier.c_str(),
            nslbdpmd5.c_str());
        SendCommRankTable(subCommRankId, globalCommInfo);
    }
    hcclNslbDpCommConfig_.push_back(globalCommInfo);

    return HCCL_SUCCESS;
}

/* 有ranktable场景， 表1 赋值 */
HcclResult hcclNslbDp::SetCommInfo_RankTableExit(RankTable_t rankTable)
{
    HCCL_DEBUG("[NSLB-DP] Entry SetCommInfo for RankTable exit");
    u16 nRanks = rankTable.rankNum;
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0 || nRanks == 1) {
        return HCCL_SUCCESS;
    }
    if (nRanks > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
        HCCL_INFO("[NSLB-DP] nRanks[%u] exceeds limit[%u], skip TBL_COMM_INFO.", nRanks, NSLBDP_RANKTOTALNUM_BLOCK_FOU);
        return HCCL_SUCCESS;
    }

    NslbDpCommConfigVal globalCommInfo = {};
    // 获取通信域唯一标识
    (void)memset_s(
        hcclNslbDpGlobalRankVal_.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(hcclNslbDpGlobalRankVal_.commDesc));
    char commDesc[COMM_DESC_MAX_LENGTH] = "HCCL_WORLD_GROUP";
    s32 sRet = memcpy_s(globalCommInfo.commDesc, sizeof(globalCommInfo.commDesc), commDesc, COMM_DESC_MAX_LENGTH);
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commDesc fail");
        return HCCL_SUCCESS;
    }

    if (IsCommDescDuplicated(globalCommInfo.commDesc, taskId)) {
        HCCL_INFO(
            "[NSLB-DP] commDesc[%s] taskId[%llu] already exists, skip TBL_COMM_INFO.", globalCommInfo.commDesc, taskId);
        return HCCL_SUCCESS;
    }
    globalCommInfo.taskId = taskId;
    globalCommInfo.rankTotalNum = nRanks;

    u64 utime
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
              .count();
    globalCommInfo.commInitTime = utime;

    u32 size = rankTable.rankList.size();
    HCCL_INFO("[NSLB-DP] RankTableExit size:[%u] success.", size);

    for (size_t ranksize = 0; ranksize < rankTable.rankList.size(); ranksize++) {
        for (size_t rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
            u16 podId = 0;
            if (rankTable.rankList[rankIndex].superPodIdx != INVALID_UINT) {
                podId = rankTable.rankList[rankIndex].superPodIdx;
            }
            // 按照rankid 排序
            if (ranksize == rankTable.rankList[rankIndex].rankId) {
                NslbDpRankInfo dpRankInfo;
                HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
                std::string deviceIp = tmpIp.GetReadableAddress();
                dpRankInfo.deviceIp = ipToUint32(deviceIp);
                HCCL_INFO("[NSLB-DP] SetCommInfo_RankTableExit deviceIp:[%s] success.", deviceIp.c_str());
                std::string serverIp = rankTable.rankList[rankIndex].serverId;
                dpRankInfo.serverIp = ServerIdToIp(serverIp);
                HCCL_INFO("[NSLB-DP] SetCommInfo_RankTableExit serverIp:[%s] success.", serverIp.c_str());
                dpRankInfo.podId = podId;
                dpRankInfo.rev = 0;
                globalCommInfo.rankInfo.push_back(dpRankInfo);
            }
        }
    }
    NSLBMD5::calculateRankInfoMd5(globalCommInfo.rankInfo, globalCommInfo.commMd5Sum);
    std::string nslbdpmd5 = NSLBMD5::md5ToString(globalCommInfo.commMd5Sum);
    HCCL_INFO("[NSLB-DP] check pmd5:[%s] success.", nslbdpmd5.c_str());

    hcclNslbDpCommConfig_.push_back(globalCommInfo);
    HCCL_DEBUG("[NSLB-DP] entry SetCommInfo_RankTableExit end");

    return HCCL_SUCCESS;
}

/* 有ranktable场景，表4赋值 */
HcclResult hcclNslbDp::SetGlobalRank_RankTableExit(const hccl::RankTable_t rankTable)
{
    u64 taskId = GetGlobalCommTaskId();
    HCCL_INFO("[NSLB-DP] set TBL_RANK for RankTableExit[%ull] success.", taskId);

    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    u16 nRanks = rankTable.rankNum;
    hcclNslbDpGlobalRankVal_.taskId = taskId;
    (void)memset_s(
        hcclNslbDpGlobalRankVal_.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(hcclNslbDpGlobalRankVal_.commDesc));
    char commDesc[COMM_DESC_MAX_LENGTH] = "HCCL_WORLD_GROUP";
    s32 sRet = memcpy_s(
        hcclNslbDpGlobalRankVal_.commDesc, sizeof(hcclNslbDpGlobalRankVal_.commDesc), commDesc, COMM_DESC_MAX_LENGTH);
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commDesc fail");
        return HCCL_SUCCESS;
    }

    hcclNslbDpGlobalRankVal_.commInitTime
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
              .count();
    hcclNslbDpGlobalRankVal_.rankTotalNum = nRanks;

    for (size_t ranksize = 0; ranksize < rankTable.rankList.size(); ranksize++) {
        for (size_t rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
            if (ranksize == rankTable.rankList[rankIndex].rankId) {
                TableFourRankInfo dpGloRankInfo;
                HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
                std::string deviceIp = tmpIp.GetReadableAddress();
                dpGloRankInfo.deviceIp = ipToUint32(deviceIp);

                std::string serverIp = rankTable.rankList[rankIndex].serverId;
                dpGloRankInfo.serverIp = ServerIdToIp(serverIp);

                hcclNslbDpGlobalRankVal_.rankInfo.push_back(dpGloRankInfo);
            }
        }
    }
    NSLBMD5::calculateTableFourRankInfoMd5(hcclNslbDpGlobalRankVal_.rankInfo, hcclNslbDpGlobalRankVal_.commMd5Sum);
    std::string nslbdpmd5 = NSLBMD5::md5ToString(hcclNslbDpGlobalRankVal_.commMd5Sum);
    HCCL_INFO("[NSLB-DP] check pmd5:[%s] success.", nslbdpmd5.c_str());
    return HCCL_SUCCESS;
}

/* 拼接 l4SPortId */
HcclResult hcclNslbDp::GetNslbDpl4SPortId(u32 rankSize, u8 algType, u16* l4SPortId)
{
    u16 priFlag = NSLBDP_PRIVATE_PORT;
    u16 CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_BEGIN;
    if (rankSize > NSLBDP_COMMINTERVAL_FLAGSIX) {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_SEV;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGFIV) {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_SIX;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGFOU) {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_FIV;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGTHR) {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_FOR;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGSEC) {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_THR;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAG) {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_SEC;
    } else {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_FIR;
    }
    u16 CommPrecisely = rankSize % NSLBDP_COMMINTERVAL_FLAG;
    u8 CommalgType = algType;

    *l4SPortId = (priFlag << NSLBDP_RANGE_ID) + (CommIntervalFlag << NSLBDP_COMMON_RANGE)
                 + (CommPrecisely << NSLBDP_ALGO_RANGE) + CommalgType;
    hcclNslbDpL4SPortId_ = *l4SPortId;

    HCCL_INFO("[NSLB-DP-L4PORT] get hcclNslbDpL4SPortId_[%u] success", hcclNslbDpL4SPortId_);
    return HCCL_SUCCESS;
}

/* 表6赋值 */
HcclResult hcclNslbDp::SetNslbDpRootRank(HcclCMDType opType, u32 rootRank, std::string identifier, u8 algType)
{
    HCCL_DEBUG("[NSLB-DP] try to collect NSLBDP_TYPE_TBL_ROOT_RANK");
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    if (hcclNslbDpRootRankVal_.taskId == 0) {
        HCCL_DEBUG("[NSLB-DP-BEGIN] RootRank first entry");
        // 获取task id
        hcclNslbDpRootRankVal_.taskId = taskId;
        s32 sRet = memset_s(
            hcclNslbDpRootRankVal_.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(hcclNslbDpRootRankVal_.commDesc));
        if (sRet != EOK) {
            HCCL_ERROR("memset_s commDesc fail");
            return HCCL_SUCCESS;
        }
        // 获取通信域唯一标识
        s32 ret
            = strncpy_s(hcclNslbDpRootRankVal_.commDesc, COMM_DESC_MAX_LENGTH, identifier.c_str(), identifier.size());
        if (ret != EOK) {
            HCCL_INFO("strncpy_s hcclNslbDpRootRankVal_.commDesc fail");
            return HCCL_SUCCESS;
        }
        hcclNslbDpRootRankVal_.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

        // commInitTime在有ranktable的赋值
        hcclNslbDpRootRankVal_.commInitTime
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                  .count();

        // 获取operator、algorithm
        hcclNslbDpRootRankVal_.oper = GetNslbOpType(opType);
        hcclNslbDpRootRankVal_.algorithm = algType;
        hcclNslbDpRootRankVal_.rootRankNum = hcclNslbDpRootRankVal_.rootRankNum + 1;

        NslbDpRankId rootRankId;
        rootRankId.rankID = rootRank;
        hcclNslbDpRootRankVal_.rankId.push_back(rootRankId);

        HCCL_INFO(
            "[NSLB-DP] Rootable rootRankNum[%u]-rootRank[%u]-algorithm[%u].", hcclNslbDpRootRankVal_.rootRankNum,
            rootRank, hcclNslbDpRootRankVal_.algorithm);

        HCCL_DEBUG("[NSLB-DP] Rootabl entry SetNslbDpRootRank end");
        SendRootRankTable();
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[NSLB-DP] Rootable rankId.size[%zu]-rootRankNum[%u]-rootRank[%u].", hcclNslbDpRootRankVal_.rankId.size(),
        hcclNslbDpRootRankVal_.rootRankNum, rootRank);

    for (const auto& rank : hcclNslbDpRootRankVal_.rankId) {
        HCCL_INFO("[NSLB-DP] Rootable rootRank exit rankID[%u]-rootRank[%u]", rank.rankID, rootRank);
        if (rank.rankID == rootRank) {
            return HCCL_SUCCESS;
        }
    }

    NslbDpRankId rootRankId;
    rootRankId.rankID = static_cast<u16>(rootRank);
    hcclNslbDpRootRankVal_.rankId.push_back(rootRankId);
    hcclNslbDpRootRankVal_.rootRankNum = hcclNslbDpRootRankVal_.rootRankNum + 1;

    // 新增场景下走send 流程
    SendRootRankTable();

    return HCCL_SUCCESS;
}

/* AlgType 的转换 */
u8 hcclNslbDp::GetNslbLevel1AlgType(AlgTypeLevel1 algValue)
{
    HCCL_DEBUG("[NSLB-DP] try to switch Level1 type to nslbtype");
    switch (algValue) {
        case AlgTypeLevel1::ALG_LEVEL1_RING: {
            return NSLB_ALGO_TYPE_RING;
        }
        case AlgTypeLevel1::ALG_LEVEL1_PIPELINE: {
            return NSLB_ALGO_TYPE_PIPELINE;
        }
        case AlgTypeLevel1::ALG_LEVEL1_HD: {
            return NSLB_ALGO_TYPE_HDR;
        }
        case AlgTypeLevel1::ALG_LEVEL1_NHR: {
            return NSLB_ALGO_TYPE_NHR;
        }
        case AlgTypeLevel1::ALG_LEVEL1_NHR_V1: {
            return NSLB_ALGO_TYPE_NHR_V1;
        }
        case AlgTypeLevel1::ALG_LEVEL1_NB: {
            return NSLB_ALGO_TYPE_NB;
        }
        case AlgTypeLevel1::ALG_LEVEL1_AHC: {
            return NSLB_ALGO_TYPE_AHC;
        }
        case AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE: {
            return NSLB_ALGO_TYPE_AHC;
        }
        default:
            return NSLB_ALGO_TYPE_NA;
    }
}

/* Level2 的 AlgType 转换 */
u8 hcclNslbDp::GetNslbLevel2AlgType(AlgTypeLevel2 algValue)
{
    HCCL_DEBUG("[NSLB-DP] try to switch Level2 type to nslbtype");
    switch (algValue) {
        case AlgTypeLevel2::ALG_LEVEL2_RING: {
            return NSLB_ALGO_TYPE_RING;
        }
        case AlgTypeLevel2::ALG_LEVEL2_HD: {
            return NSLB_ALGO_TYPE_HDR;
        }
        case AlgTypeLevel2::ALG_LEVEL2_NHR: {
            return NSLB_ALGO_TYPE_NHR;
        }
        case AlgTypeLevel2::ALG_LEVEL2_NB: {
            return NSLB_ALGO_TYPE_NB;
        }
        case AlgTypeLevel2::ALG_LEVEL2_PIPELINE: {
            return NSLB_ALGO_TYPE_PIPELINE;
        }
        default:
            return NSLB_ALGO_TYPE_NA;
    }
}

/* opType 的转换 */
u8 hcclNslbDp::GetNslbOpType(HcclCMDType opType)
{
    HCCL_DEBUG("[NSLB-DP] try to get nslb optype");
    switch (opType) {
        case HcclCMDType::HCCL_CMD_BROADCAST: {
            return NSLBDP_CMD_BROADCAST;
        }
        case HcclCMDType::HCCL_CMD_ALLREDUCE: {
            return NSLBDP_CMD_ALLREDUCE;
        }
        case HcclCMDType::HCCL_CMD_REDUCE: {
            return NSLBDP_CMD_REDUCE;
        }
        case HcclCMDType::HCCL_CMD_SEND: {
            return NSLBDP_CMD_BATCH_SEND_RECV;
        }
        case HcclCMDType::HCCL_CMD_RECEIVE: {
            return NSLBDP_CMD_BATCH_SEND_RECV;
        }
        case HcclCMDType::HCCL_CMD_ALLGATHER: {
            return NSLBDP_CMD_ALLGATHER;
        }
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER: {
            return NSLBDP_CMD_REDUCE_SCATTER;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALLV: {
            return NSLBDP_CMD_ALLTOALLV;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALLVC: {
            return NSLBDP_CMD_ALLTOALLVC;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALL: {
            return NSLBDP_CMD_ALLTOALL;
        }
        case HcclCMDType::HCCL_CMD_GATHER: {
            return NSLBDP_CMD_GATHER;
        }
        case HcclCMDType::HCCL_CMD_SCATTER: {
            return NSLBDP_CMD_SCATTER;
        }
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV: {
            return NSLBDP_CMD_BATCH_SEND_RECV;
        }
        default:
            return 0;
    }
}

/* 获取通信量的前4bit */
u64 hcclNslbDp::GetNslbDpFirstFourBit(u8 opType, u8 algType)
{
    u64 firstFourBit = 0;
    if (opType == NSLBDP_CMD_ALLREDUCE) {
        firstFourBit = 1 << NSLBDP_BEGINFOURBIT;
        firstFourBit = firstFourBit + 1;
    }

    if (opType == NSLBDP_CMD_ALLGATHER) {
        firstFourBit = 1 << 1;
    }

    HCCL_INFO("[NSLB-DP-FIRST4] try to get first FourBit[%llu]-algType[%u]", firstFourBit, algType);
    return firstFourBit;
}

/* 校验算法的一致性 */
bool hcclNslbDp::CheckAlgoConsistency(HcclCMDType opType, std::string& algName)
{
    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        if (algName.find("AllReduce") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        if (algName.find("AllGather") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        if (algName.find("ReduceScatter") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        if (algName != "RunAlltoAllVFullMesh") {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        if (algName.find("BroadCast") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_SCATTER) {
        if (algName.find("Scatter") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    }
    return true;
}

bool hcclNslbDp::CheckSupportOptype(HcclCMDType opType)
{
    HCCL_DEBUG("[NSLB-DP] try to check support optype");
    if (GetNslbOpType(opType) != NSLBDP_CMD_ALLREDUCE && GetNslbOpType(opType) != NSLBDP_CMD_ALLGATHER
        && GetNslbOpType(opType) != NSLBDP_CMD_REDUCE_SCATTER && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALL
        && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALLV && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALLVC
        && GetNslbOpType(opType) != NSLBDP_CMD_BROADCAST && GetNslbOpType(opType) != NSLBDP_CMD_SCATTER
        && GetNslbOpType(opType) != NSLBDP_CMD_BATCH_SEND_RECV && GetNslbOpType(opType) != NSLBDP_CMD_REDUCE) {
        return false;
    }
    return true;
}

/* 写算法算子表--表3 */
HcclResult hcclNslbDp::GetAlgAdjacencyTable(
    HcclCMDType opType, u32 srcLocalRankId, u32 rootRank, u8 algType, std::string identifier, AdjInfo nslbAdjInfo)
{
    HCCL_DEBUG("[NSLB-DP] check table NSLBDP_TYPE_TBL_ADJ size:[%zu].", hcclNslbDpAlgorithmInfo_.size());
    if (CheckAhcSupport(algType, identifier) == false) {
        HCCL_RUN_INFO("[NSLB-DP-ADJ] Check AHC commoninfo is not support.");
        return HCCL_SUCCESS;
    }
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    if (CheckSupportOptype(opType) == false) {
        HCCL_INFO("[NSLB-DP-OPER] CheckSupportOptype false .");
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[NSLB-DP-ADJ] opType:[%u],srcLocalRankId[%u],rootRank[%u]-commDesc[%s],dstRankNum:[%u].", opType,
        srcLocalRankId, rootRank, identifier.c_str(), nslbAdjInfo.dstRankNum);
    /* A3场景SEND/RESV/BATCH_SEND_RECV算子特殊处理：从表1获取rankTotalNum/podIdZeroNum，模拟双路邻接信息 */
    if (GetNslbOpType(opType) == HCCL_CMD_BATCH_SEND_RECV && nslbAdjInfo.dstRankNum == 0) {
        if (!TryFillA3SimulatedAdjInfo(identifier, srcLocalRankId, nslbAdjInfo)) {
            HCCL_RUN_INFO("[NSLB-DP-ADJ] A3 simulated adj info skipped for identifier[%s].", identifier.c_str());
            return HCCL_SUCCESS;
        }
    }

    NslbDpAlgorithmInfo algorithmInfo;
    algorithmInfo.taskId = taskId;
    if (InitAlgInfoCommDesc(algorithmInfo, identifier) == false) {
        return HCCL_SUCCESS;
    }
    if (FillAlgInfoCommMd5(algorithmInfo) == false) {
        return HCCL_SUCCESS;
    }
    FillAlgInfoBaseFields(algorithmInfo, opType, srcLocalRankId, rootRank, algType);

    HCCL_RUN_INFO(
        "[NSLB-DP-ADJ] add adjINfo:identifier[%s],[%llu],[%u],[%u],[%zu] success.", identifier.c_str(), taskId,
        srcLocalRankId, rootRank, GetNslbOpType(opType), algType, nslbAdjInfo.nsAdjInfo.size());

    if (IsAlgAdjacencyDuplicated(algorithmInfo) == true) {
        HCCL_INFO("[NSLB-DP] Deduplication hcclNslbDpAlgorithmInfo_");
        return HCCL_SUCCESS;
    }

    if (FillAlgInfoAdjInfo(algorithmInfo, nslbAdjInfo, srcLocalRankId) == false) {
        return HCCL_SUCCESS;
    }
    algorithmInfo.sedFlag = 0;
    hcclNslbDpAlgorithmInfo_.push_back(algorithmInfo);
    HCCL_DEBUG("[NSLB-DP] entry GetAlgAdjacencyTable end");
    return HCCL_SUCCESS;
}

/* 按identifier取表1MD5，再取同MD5中commInitTime最小的commDesc；失败返回false */
bool hcclNslbDp::FindMinInitTimeCommDesc(const std::string& identifier, std::string& commDesc, u64& minInitTime)
{
    // 基于 identifier 查找通信域表1中对应的条目，获取其 MD5 值
    u8 targetMd5[sizeof(NslbDpCommConfigVal::commMd5Sum)];
    bool found = false;
    for (const auto& info : hcclNslbDpCommConfig_) {
        if (strcmp(info.commDesc, identifier.c_str()) == 0) {
            s32 sRet = memcpy_s(targetMd5, sizeof(targetMd5), info.commMd5Sum, sizeof(info.commMd5Sum));
            if (sRet != EOK) {
                HCCL_ERROR("[NSLB-DP] memcpy_s commMd5Sum fail");
                return false;
            }
            found = true;
            break;
        }
    }
    if (!found) {
        HCCL_INFO("[NSLB-DP] identifier[%s] not found in hcclNslbDpCommConfig_, skip.", identifier.c_str());
        return false;
    }

    // 基于 MD5 值遍历通信域表1，获取 commInitTime 最小的条目对应的 commDesc
    bool firstMatch = true;
    minInitTime = 0;
    const char* minCommDesc = nullptr;
    for (const auto& info : hcclNslbDpCommConfig_) {
        if (memcmp(info.commMd5Sum, targetMd5, sizeof(targetMd5)) == 0) {
            if (firstMatch || info.commInitTime < minInitTime) {
                minInitTime = info.commInitTime;
                minCommDesc = info.commDesc;
                firstMatch = false;
            }
        }
    }
    if (minCommDesc == nullptr) {
        HCCL_INFO("[NSLB-DP] no entry with same MD5 found for identifier[%s].", identifier.c_str());
        return false;
    }

    commDesc = minCommDesc;
    return true;
}

/* 命中"存在不同commDesc但MD5相同的已有条目"时返回true，用于跳过重复下发表1 */
bool hcclNslbDp::ShouldSkipSendByMd5(const NslbDpCommConfigVal& info)
{
    for (const auto& existing : hcclNslbDpCommConfig_) {
        if (strcmp(existing.commDesc, info.commDesc) != 0
            && memcmp(existing.commMd5Sum, info.commMd5Sum, sizeof(info.commMd5Sum)) == 0) {
            return true;
        }
    }
    return false;
}

/* 初始化commDesc并校验通信域是否存在；存在返回true，不存在返回false */
bool hcclNslbDp::InitAlgInfoCommDesc(NslbDpAlgorithmInfo& algorithmInfo, const std::string& identifier)
{
    std::string minCommDesc;
    u64 minInitTime = 0;
    if (FindMinInitTimeCommDesc(identifier, minCommDesc, minInitTime) == false) {
        return false;
    }

    // 使用最小 commInitTime 对应的 commDesc 赋值
    (void)memset_s(algorithmInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(algorithmInfo.commDesc));
    s32 ret = strncpy_s(algorithmInfo.commDesc, COMM_DESC_MAX_LENGTH, minCommDesc.c_str(), minCommDesc.size());
    if (ret != EOK) {
        HCCL_ERROR("[NSLB-DP] strncpy_s algorithmInfo.commDesc fail");
        return false;
    }
    algorithmInfo.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

    HCCL_INFO(
        "[NSLB-DP] InitAlgInfoCommDesc commDesc[%s] from min commInitTime[%llu].", algorithmInfo.commDesc, minInitTime);
    return true;
}

/* 根据表一信息填充commMd5Sum；memcpy失败返回false */
bool hcclNslbDp::FillAlgInfoCommMd5(NslbDpAlgorithmInfo& algorithmInfo)
{
    // 根据表一信息填充MD5
    for (size_t comsize = 0; comsize < hcclNslbDpCommConfig_.size(); comsize++) {
        if (strcmp(algorithmInfo.commDesc, hcclNslbDpCommConfig_[comsize].commDesc) == 0) {
            s32 sRet = memcpy_s(
                algorithmInfo.commMd5Sum, sizeof(algorithmInfo.commMd5Sum), hcclNslbDpCommConfig_[comsize].commMd5Sum,
                sizeof(hcclNslbDpCommConfig_[comsize].commMd5Sum));
            if (sRet != EOK) {
                HCCL_ERROR("memcpy_s commMd5Sum fail");
                return false;
            }
            break;
        }
    }
    return true;
}

/* 填充srcLocalRankId/rootRank/oper/algorithm基础字段 */
void hcclNslbDp::FillAlgInfoBaseFields(
    NslbDpAlgorithmInfo& algorithmInfo, HcclCMDType opType, u32 srcLocalRankId, u32 rootRank, u8 algType)
{
    algorithmInfo.srcLocalRankId = srcLocalRankId;
    algorithmInfo.rootRank = static_cast<u16>(rootRank);
    algorithmInfo.oper = GetNslbOpType(opType);
    algorithmInfo.algorithm = algType;
}

/* A3场景SEND/RECEIVE/BATCH_SEND_RECV算子：从表1获取rankTotalNum/podIdZeroNum，模拟双路邻接信息 */
bool hcclNslbDp::TryFillA3SimulatedAdjInfo(const std::string& identifier, u32 srcLocalRankId, AdjInfo& nslbAdjInfo)
{
    /* 遍历通信域表1，获取指定identifier的rankTotalNum以及podId==0的卡的总数 */
    u16 rankTotalNum = 0;
    u16 podIdZeroNum = 0;
    for (const auto& configInfo : hcclNslbDpCommConfig_) {
        if (strcmp(configInfo.commDesc, identifier.c_str()) == 0) {
            rankTotalNum = configInfo.rankTotalNum;
            for (const auto& rankInfo : configInfo.rankInfo) {
                if (rankInfo.podId == 0) {
                    podIdZeroNum++;
                }
            }
            HCCL_RUN_INFO(
                "[NSLB-DP-ADJ] identifier[%s] rankTotalNum[%u] podIdZeroNum[%u].", identifier.c_str(), rankTotalNum,
                podIdZeroNum);
            break;
        }
    }
    if (podIdZeroNum == 0 || rankTotalNum == 0 || podIdZeroNum >= rankTotalNum) {
        HCCL_RUN_INFO(
            "[NSLB-DP-ADJ] identifier[%s] abnormal podIdZeroNum[%u] rankTotalNum[%u], skip.", identifier.c_str(),
            podIdZeroNum, rankTotalNum);
        return false;
    }

    /* 模拟A3场景下SEND/RESV场景下的通信域 */
    NslbDpAdjInfo adjInfoStep1 = {};
    adjInfoStep1.dstLocalRankId = (srcLocalRankId + podIdZeroNum) % rankTotalNum;
    adjInfoStep1.phaseId = 1;
    adjInfoStep1.rev = 0;
    nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep1);
    /* 只有总卡数大于2时，才需要添加第二个邻接点 */
    if (rankTotalNum > 2) {
        NslbDpAdjInfo adjInfoStep2 = {};
        adjInfoStep2.dstLocalRankId = (srcLocalRankId + rankTotalNum - podIdZeroNum) % rankTotalNum;
        adjInfoStep2.phaseId = 2;
        adjInfoStep2.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep2);
    }
    /* 从实际 push_back 的数量赋值，保证 dstRankNum 与 nsAdjInfo.size() 一致 */
    nslbAdjInfo.dstRankNum = static_cast<u16>(nslbAdjInfo.nsAdjInfo.size());
    return true;
}

/* 判断是否与已有邻接表项重复（去重） */
bool hcclNslbDp::IsAlgAdjacencyDuplicated(const NslbDpAlgorithmInfo& algorithmInfo)
{
    for (const auto& info : hcclNslbDpAlgorithmInfo_) {
        HCCL_INFO(
            "[NSLB-DP-ADJ] info: *[%llu]-[%u]-[%u]-[%u]-[%u]* success.", algorithmInfo.taskId, info.srcLocalRankId,
            info.rootRank, info.oper, info.algorithm);
        if (info.taskId == algorithmInfo.taskId && info.srcLocalRankId == algorithmInfo.srcLocalRankId
            && info.rootRank == algorithmInfo.rootRank && info.oper == algorithmInfo.oper
            && info.algorithm == algorithmInfo.algorithm && strcmp(algorithmInfo.commDesc, info.commDesc) == 0) {
            return true;
        }
    }
    return false;
}

/* 填充dstRankNum与AdjInfo邻接信息；nsAdjInfo为空返回false */
bool hcclNslbDp::FillAlgInfoAdjInfo(NslbDpAlgorithmInfo& algorithmInfo, const AdjInfo& nslbAdjInfo, u32 srcLocalRankId)
{
    algorithmInfo.dstRankNum = nslbAdjInfo.dstRankNum;
    HCCL_INFO("[NSLB-DP-ADJ] nslbAdjInfo.dstRankNum:[%u].", nslbAdjInfo.dstRankNum);

    if (nslbAdjInfo.nsAdjInfo.size() == 0) {
        algorithmInfo.dstRankNum = 0;
        HCCL_INFO("[NSLB-DP] get nsAdjInfo fail dstRankNum:[%u]", algorithmInfo.dstRankNum);
        return false;
    }

    for (size_t rankIndex = 0; rankIndex < nslbAdjInfo.nsAdjInfo.size(); rankIndex++) {
        NslbDpAdjInfo adjInfo;
        adjInfo.dstLocalRankId = nslbAdjInfo.nsAdjInfo[rankIndex].dstLocalRankId;
        adjInfo.phaseId = nslbAdjInfo.nsAdjInfo[rankIndex].phaseId;
        adjInfo.rev = nslbAdjInfo.nsAdjInfo[rankIndex].rev;
        HCCL_INFO(
            "[NSLB-DP-ADJ] adjINfo:[%u]-[%u]-[%u] success.", srcLocalRankId,
            nslbAdjInfo.nsAdjInfo[rankIndex].dstLocalRankId, nslbAdjInfo.nsAdjInfo[rankIndex].phaseId);
        algorithmInfo.AdjInfo.push_back(adjInfo);
    }
    return true;
}

bool hcclNslbDp::CheckCommDescExit(NslbDpOperatorInfo& OperatorInfo)
{
    HCCL_DEBUG("[NSLB-DP-OPER] Check CommDescExit size:[%zu].", hcclNslbDpCommConfig_.size());
    for (const auto& info : hcclNslbDpCommConfig_) {
        HCCL_DEBUG("[NSLB-DP-OPER] CheckCommDescExit info.commDesc[%s] .", info.commDesc);
        if (strcmp(OperatorInfo.commDesc, info.commDesc) == 0) {
            return true;
        }
    }
    return false;
}

void hcclNslbDp::fullcommDescInitTime(std::string identifier, NslbDpOperatorInfo& OperatorInfo)
{
    std::string minCommDesc;
    u64 minInitTime = 0;
    if (FindMinInitTimeCommDesc(identifier, minCommDesc, minInitTime) == false) {
        return;
    }

    // 使用最小 commInitTime 对应的 commDesc 赋值
    s32 ret = strncpy_s(OperatorInfo.commDesc, COMM_DESC_MAX_LENGTH, minCommDesc.c_str(), minCommDesc.size());
    if (ret != EOK) {
        HCCL_INFO("[NSLB-DP] strncpy_s OperatorInfo.commDesc fail");
        return;
    }
    HCCL_DEBUG(
        "[NSLB-DP-OPER] fullcommDescInitTime commDesc[%s] from min commInitTime[%llu].", OperatorInfo.commDesc,
        minInitTime);

    OperatorInfo.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

    // commInitTime在有ranktable的赋值
    OperatorInfo.commInitTime
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
              .count();

    return;
}

bool hcclNslbDp::CheckSameOperatorVal(size_t operSize, NslbDpOperatorInfo& OperatorInfo, u32 rootRank)
{
    size_t num = hcclNslbDpOperatorVal_.size();
    if (operSize >= num) {
        return false;
    }
    if (hcclNslbDpOperatorVal_[operSize].taskId == OperatorInfo.taskId
        && hcclNslbDpOperatorVal_[operSize].rootRank == rootRank
        && hcclNslbDpOperatorVal_[operSize].oper == OperatorInfo.oper
        && hcclNslbDpOperatorVal_[operSize].algorithm == OperatorInfo.algorithm
        && strcmp(OperatorInfo.commDesc, hcclNslbDpOperatorVal_[operSize].commDesc) == 0) {
        return true;
    }
    return false;
}

/* 写算法算子表--表2 */
HcclResult hcclNslbDp::GenerateOpAndAdjTable(
    HcclCMDType opType, u32 rootRank, u32 srcLocalRankId, u8 algType, std::string identifier, u64 count, u32 rankSize)
{
    HCCL_INFO("[NSLB-DP-OPER] count=[%llu], hcclNslbDpOperatorVal_ size:[%zu].", count, hcclNslbDpOperatorVal_.size());
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }

    // 获取，0号
    if (rootRank != 0) {
        return HCCL_SUCCESS;
    }
    u64 trafficNum = count;
    if (CheckSupportOptype(opType) == false) {
        trafficNum = 0;
        HCCL_DEBUG("[NSLB-DP-OPER] CheckSupportOptype false trafficNum == 0.");
    }

    NslbDpOperatorInfo OperatorInfo = {};
    // 获取task id
    OperatorInfo.taskId = taskId;
    (void)memset_s(OperatorInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(OperatorInfo.commDesc));

    fullcommDescInitTime(identifier, OperatorInfo);

    // 去除不存在的通信域信息
    if (CheckCommDescExit(OperatorInfo) == false) {
        HCCL_INFO("[NSLB-DP-OPER] CheckCommDesc not exit ");
        return HCCL_SUCCESS;
    }

    // 获取operator、algorithm
    OperatorInfo.oper = GetNslbOpType(opType);
    OperatorInfo.algorithm = algType;
    u64 trafficCount = GetNslbDpFirstFourBit(OperatorInfo.oper, OperatorInfo.algorithm);
    trafficCount = (trafficCount << NSLBDP_TRAFFICCONUT) + trafficNum;

    for (size_t operSize = 0; operSize < hcclNslbDpOperatorVal_.size(); operSize++) {
        if (CheckSameOperatorVal(operSize, OperatorInfo, rootRank) == true) {
            if (hcclNslbDpOperatorVal_[operSize].trafficCnt < trafficCount && srcLocalRankId == 0) {
                hcclNslbDpOperatorVal_[operSize].trafficCnt = trafficCount;
                hcclNslbDpOperatorVal_[operSize].sedFlag = 0;
                SendRankTableOpAndAdj(hcclNslbDpOperatorVal_[operSize]);
                HCCL_RUN_INFO(
                    "[NSLB-DP-ADJ] commDesc[%s] try to update trafficCnt[%llu] success.", OperatorInfo.commDesc,
                    trafficCount);
            }
            return HCCL_SUCCESS;
        }
    }

    OperatorInfo.trafficCnt = trafficCount; // 判断变大
    OperatorInfo.rootRank = rootRank;
    HCCL_INFO(
        "[NSLB-DP-OPER] add operInfo:***[%llu]***[%llu]***[%u]***[%u]***[%u] success.", taskId,
        OperatorInfo.commInitTime, rootRank, OperatorInfo.oper, OperatorInfo.algorithm);

    GetNslbDpl4SPortId(rankSize, algType, &OperatorInfo.l4SPortId);
    if (srcLocalRankId == 0) {
        SendRankTableOpAndAdj(OperatorInfo);
    }
    hcclNslbDpOperatorVal_.push_back(OperatorInfo);

    return HCCL_SUCCESS;
}

/* 根将表1 序列化处理 */
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableFir(NslbDpCommConfigInfo cominfo)
{
    HCCL_DEBUG("[NSLB-DP] entry serializeTLV TBL_COMM_INFO beg");

    std::vector<uint8_t> tlvData;

    // Task ID
    uint64_t netTaskId = htobe64(cominfo.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // CommDesc
    tlvData.insert(tlvData.end(), cominfo.commDesc, cominfo.commDesc + COMM_DESC_MAX_LENGTH);

    // CommInitTime
    uint64_t netCommInitTime = htobe64(cominfo.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime,
        sizeof(netCommInitTime));

    // packetId
    uint16_t netPacketId = htons(cominfo.packetId);
    tlvData.resize(tlvData.size() + sizeof(netPacketId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netPacketId), sizeof(netPacketId), &netPacketId, sizeof(netPacketId));

    // rev
    uint16_t netRev = htons(cominfo.rev);
    tlvData.resize(tlvData.size() + sizeof(netRev));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev), sizeof(netRev), &netRev, sizeof(netRev));

    // packetNum
    uint16_t netPacketNum = htons(cominfo.packetNum);
    tlvData.resize(tlvData.size() + sizeof(netPacketNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netPacketNum), sizeof(netPacketNum), &netPacketNum,
        sizeof(netPacketNum));

    // revSecond
    uint16_t netRevSecond = htons(cominfo.revSecond);
    tlvData.resize(tlvData.size() + sizeof(netRevSecond));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRevSecond), sizeof(netRevSecond), &netRevSecond,
        sizeof(netRevSecond));

    // sendRankInfo
    for (const auto& rankInfo : cominfo.sendRankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp,
            sizeof(netDeviceIp));
    }

    // CommMd5Sum
    tlvData.insert(tlvData.end(), cominfo.commMd5Sum, cominfo.commMd5Sum + sizeof(cominfo.commMd5Sum));

    // RankTotalNum
    uint16_t netRankTotalNum = htons(cominfo.rankTotalNum);
    tlvData.resize(tlvData.size() + sizeof(netRankTotalNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRankTotalNum), sizeof(netRankTotalNum), &netRankTotalNum,
        sizeof(netRankTotalNum));

    // Rank Number
    uint16_t netRankNum = htons(cominfo.rankNum);
    tlvData.resize(tlvData.size() + sizeof(netRankNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRankNum), sizeof(netRankNum), &netRankNum, sizeof(netRankNum));

    // Rank Info
    for (const auto& rankInfo : cominfo.rankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        uint32_t netServerIp = htonl(rankInfo.serverIp);
        uint16_t netPodId = htons(rankInfo.podId);
        uint16_t netRev3 = htons(rankInfo.rev);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp,
            sizeof(netDeviceIp));
        tlvData.resize(tlvData.size() + sizeof(netServerIp));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netServerIp), sizeof(netServerIp), &netServerIp,
            sizeof(netServerIp));
        tlvData.resize(tlvData.size() + sizeof(netPodId));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netPodId), sizeof(netPodId), &netPodId, sizeof(netPodId));
        tlvData.resize(tlvData.size() + sizeof(netRev3));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev3), sizeof(netRev3), &netRev3, sizeof(netRev3));
    }
    HCCL_DEBUG("[NSLB-DP] entry serializeTLV TBL_COMM_INFO end");

    return tlvData;
}

void hcclNslbDp::fullCommConfigInfo(NslbDpCommConfigInfo& tab_f, NslbDpCommConfigVal cominfo, u32 packetNum)
{
    HCCL_DEBUG("[NSLB-DP] entry full CommConfigInfo");
    tab_f.taskId = cominfo.taskId;
    s32 sRet = memcpy_s(tab_f.commDesc, sizeof(tab_f.commDesc), cominfo.commDesc, COMM_DESC_MAX_LENGTH);
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s commDesc fail");
    }
    tab_f.commInitTime = cominfo.commInitTime;

    tab_f.rev = 0;
    tab_f.packetNum = packetNum;
    tab_f.revSecond = 0;
    u32 sendCnt = std::min(packetNum, static_cast<u32>(cominfo.rankInfo.size()));
    for (u32 dip = 0; dip < sendCnt; dip++) {
        tab_f.sendRankInfo[dip].deviceIp = cominfo.rankInfo[dip].deviceIp;
    }
    sRet = memcpy_s(tab_f.commMd5Sum, sizeof(tab_f.commMd5Sum), cominfo.commMd5Sum, sizeof(cominfo.commMd5Sum));
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s commmd5 fail");
    }
}

/* 根据表1 中的rank数量进行分片处理 */
HcclResult hcclNslbDp::SendTableProc(u32 rank, u32 packetNum, NslbDpCommConfigVal cominfo)
{
    if (packetNum - 1 < rank) {
        HCCL_INFO("[NSLB-DP] rank [%u] no need send TableFir(packetNum[%u]).", rank, packetNum);
        return HCCL_SUCCESS;
    }
    u32 packetIndex = rank;
    NslbDpCommConfigInfo tab_f = {};
    fullCommConfigInfo(tab_f, cominfo, packetNum);
    tab_f.packetId = packetIndex;

    u32 start = packetIndex * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    u32 end = (packetIndex + 1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    u32 totalSize = static_cast<u32>(cominfo.rankInfo.size());
    if (totalSize < start) {
        HCCL_INFO("[NSLB-DP] Comm RankInfo not as expected, size[%u] < start[%u]", totalSize, start);
        return HCCL_SUCCESS;
    }
    end = std::min(end, totalSize);

    u32 count = end - start;
    tab_f.rankInfo.reserve(count);
    for (u32 i = start; i < end; i++) {
        tab_f.rankInfo.push_back(cominfo.rankInfo[i]);
    }
    tab_f.rankTotalNum = cominfo.rankTotalNum;
    tab_f.rankNum = static_cast<u16>(tab_f.rankInfo.size());

    const char* tag = (packetIndex != packetNum - 1) ? "SendTableProc-F" : "SendTableProc-N";
    HCCL_INFO(
        "[NSLB-DP] %s info:[%u]-[%u]-[%u]-[%u].", tag, packetNum, packetIndex, cominfo.rankTotalNum,
        tab_f.rankInfo.size());
    HCCL_DEBUG(
        "[NSLB-DP] SendRankTable-info:[%u]-[%u]-[%u]-[%u].", tab_f.rankTotalNum, tab_f.packetNum, tab_f.rankTotalNum,
        tab_f.rankInfo.size());
    SendRankTable(tab_f);

    return HCCL_SUCCESS;
}

/* 遍历表1 执行send 流程 */
HcclResult hcclNslbDp::SendTableFir(uint32_t rank)
{
    size_t size = hcclNslbDpCommConfig_.size();
    HCCL_INFO("[NSLB-DP] SendTableFir size:[%u] success.", size);
    for (size_t i = 0; i < hcclNslbDpCommConfig_.size(); i++) {
        u32 rankTotalNum = hcclNslbDpCommConfig_[i].rankTotalNum;
        if (rankTotalNum > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
            return HCCL_SUCCESS;
        }
        u32 packetNum = CalcPacketNum(rankTotalNum);
        SendTableProc(rank, packetNum, hcclNslbDpCommConfig_[i]);
    }
    HCCL_DEBUG("[NSLB-DP] entry SendTableFir end");
    return HCCL_SUCCESS;
}

HcclResult hcclNslbDp::SendRankTable(NslbDpCommConfigInfo tab_f)
{
    if (GetInitNetCoFlag() == false) {
        return HCCL_SUCCESS;
    }

    u32 tablen = sizeof(tab_f);
    HCCL_INFO("[NSLB-DP] SendRankTable NslbDpCommConfigInfo len:[%u].", tablen);

    std::vector<uint8_t> tlvData = serializeTLV_TableFir(tab_f);
    u32 datlen = tlvData.size();
    HCCL_INFO("[NSLB-DP] SendRankTable tlvData.len:[%u] success.", datlen);

    if (nslbdp_handle_ == nullptr) {
        HCCL_INFO("[NSLB-DP] ndlbdp nslbdp_handle_ error SendRankTable.");
        return HCCL_SUCCESS;
    }

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_COMM_INFO;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        nslbdp_handle_, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));

    HCCL_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_TBL_COMM_INFO(1001) to hccp. ret(%d)\n", ret);
    return HCCL_SUCCESS;
}

/* 根将表2 序列化处理 */
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableOpAndAdj(NslbDpOperatorInfo& info)
{
    HCCL_DEBUG("[NSLB-DP] ndlbdp entry serializeTLV TableOpAndAdj.");
    std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), info.commDesc, info.commDesc + COMM_DESC_MAX_LENGTH);

    // 处理 commInitTime
    uint64_t netCommInitTime = htobe64(info.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime,
        sizeof(netCommInitTime));

    // 处理 oper
    tlvData.push_back(info.oper);

    // 处理 algorithm
    tlvData.push_back(info.algorithm);

    // 处理 rootRank
    uint16_t netRootRank = htons(info.rootRank);
    tlvData.resize(tlvData.size() + sizeof(netRootRank));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRootRank), sizeof(netRootRank), &netRootRank, sizeof(netRootRank));

    // 处理 trafficCnt
    uint64_t netTrafficCnt = htobe64(info.trafficCnt);
    tlvData.resize(tlvData.size() + sizeof(netTrafficCnt));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netTrafficCnt), sizeof(netTrafficCnt), &netTrafficCnt,
        sizeof(netTrafficCnt));

    // 处理 l4SPortId
    uint16_t netL4SPortId = htons(info.l4SPortId);
    tlvData.resize(tlvData.size() + sizeof(netL4SPortId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netL4SPortId), sizeof(netL4SPortId), &netL4SPortId,
        sizeof(netL4SPortId));

    // 处理 maskLen
    uint16_t netMaskLen = htons(info.maskLen);
    tlvData.resize(tlvData.size() + sizeof(netMaskLen));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netMaskLen), sizeof(netMaskLen), &netMaskLen, sizeof(netMaskLen));
    HCCL_INFO("[NSLB-DP] entry serializeTLV_TableOpAndAdj end.");
    return tlvData;
}

/* 执行send 流程 */
HcclResult hcclNslbDp::SendRankTableOpAndAdj(NslbDpOperatorInfo& tab_f)
{
    if (GetInitNetCoFlag() == false) {
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[NSLB-DP] ndlbdp entry SendRankTableOpAndAdj.");
    std::vector<uint8_t> tlvData = serializeTLV_TableOpAndAdj(tab_f);

    u32 datlen = tlvData.size();
    HCCL_INFO("[NSLB-DP] SendRankTableOpAndAdj tlvData.len:[%u] success.", datlen);

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_OPER;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        nslbdp_handle_, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));

    HCCL_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_TBL_OPER(1002) to hccp. ret(%d)\n", ret);

    return HCCL_SUCCESS;
}

/* 遍历表2 执行send 流程 */
HcclResult hcclNslbDp::SendOpAndAdjTable()
{
    HCCL_DEBUG("[NSLB-DP] ndlbdp entry SendOpAndAdjTable.");
    for (size_t i = 0; i < hcclNslbDpOperatorVal_.size(); i++) {
        if (hcclNslbDpOperatorVal_[i].sedFlag == 1) {
            continue;
        }
        NslbDpOperatorInfo tab_f = {};
        tab_f.taskId = hcclNslbDpOperatorVal_[i].taskId;
        s32 sRet = memcpy_s(
            tab_f.commDesc, sizeof(tab_f.commDesc), hcclNslbDpOperatorVal_[i].commDesc, COMM_DESC_MAX_LENGTH);
        if (sRet != EOK) {
            HCCL_ERROR("memcpy_s commDesc fail");
        }
        tab_f.commInitTime = hcclNslbDpOperatorVal_[i].commInitTime;
        tab_f.rootRank = hcclNslbDpOperatorVal_[i].rootRank;
        tab_f.oper = hcclNslbDpOperatorVal_[i].oper;
        tab_f.algorithm = hcclNslbDpOperatorVal_[i].algorithm;
        tab_f.trafficCnt = hcclNslbDpOperatorVal_[i].trafficCnt;
        tab_f.l4SPortId = hcclNslbDpOperatorVal_[i].l4SPortId;
        tab_f.maskLen = hcclNslbDpOperatorVal_[i].maskLen;
        SendRankTableOpAndAdj(tab_f);
        hcclNslbDpOperatorVal_[i].sedFlag = 1;
        HCCL_INFO("[NSLB-DP] try to sen RankTableOpAndAdj times:[%u].", i);
    }
    HCCL_INFO("[NSLB-DP] SendOpAndAdjTable end.");
    return HCCL_SUCCESS;
}

/* 根将表3 序列化处理 */
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableAlgorithmInfo(NslbDpAlgorithmTlv& info)
{
    HCCL_DEBUG("[NSLB-DP] entry serializeTLV_TableAlgorithmInfo.");
    std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), info.commDesc, info.commDesc + COMM_DESC_MAX_LENGTH);

    // CommMd5Sum
    tlvData.insert(tlvData.end(), info.commMd5Sum, info.commMd5Sum + sizeof(info.commMd5Sum));

    // 处理 srcLocalRankId
    uint16_t netSrcLocalRankId = htons(info.srcLocalRankId);
    tlvData.resize(tlvData.size() + sizeof(netSrcLocalRankId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netSrcLocalRankId), sizeof(netSrcLocalRankId), &netSrcLocalRankId,
        sizeof(netSrcLocalRankId));

    // 处理 oper
    tlvData.push_back(info.oper);

    // 处理 algorithm
    tlvData.push_back(info.algorithm);

    // 处理 rootRank
    uint16_t netRootRank = htons(info.rootRank);
    tlvData.resize(tlvData.size() + sizeof(netRootRank));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRootRank), sizeof(netRootRank), &netRootRank, sizeof(netRootRank));

    // 处理 rev
    uint16_t netRev = htons(info.rev);
    tlvData.resize(tlvData.size() + sizeof(netRev));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev), sizeof(netRev), &netRev, sizeof(netRev));

    // 处理 dstRankNum
    uint16_t netDstRankNum = htons(info.dstRankNum);
    tlvData.resize(tlvData.size() + sizeof(netDstRankNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netDstRankNum), sizeof(netDstRankNum), &netDstRankNum,
        sizeof(netDstRankNum));

    // 处理 revSecond
    uint16_t netRevSecond = htons(info.revsecond);
    tlvData.resize(tlvData.size() + sizeof(netRevSecond));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRevSecond), sizeof(netRevSecond), &netRevSecond,
        sizeof(netRevSecond));

    HCCL_INFO("[NSLB-DP] serializeTLV_TableAlgorithmInfo proc adjInfo.");
    // 处理 adjInfo
    for (const auto& adj : info.AdjInfo) {
        uint16_t netDstLocalRankId = htons(adj.dstLocalRankId);
        tlvData.resize(tlvData.size() + sizeof(netDstLocalRankId));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netDstLocalRankId), sizeof(netDstLocalRankId), &netDstLocalRankId,
            sizeof(netDstLocalRankId));
        tlvData.push_back(adj.phaseId);
        tlvData.push_back(adj.rev);
        HCCL_INFO("[NSLB-DP] serializeTLV_TableAlgo adjInfo:***[%u]***[%c]***.", adj.dstLocalRankId, adj.phaseId);
    }
    HCCL_DEBUG("[NSLB-DP] serializeTLV_TableAlgo proc adjInfo end.");

    return tlvData;
}

HcclResult hcclNslbDp::SendRankTableAlgorithmInfo(NslbDpAlgorithmTlv& tab_f)
{
    if (GetInitNetCoFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableAlgorithmInfo(tab_f);

    u32 datlen = tlvData.size();
    HCCL_INFO("[NSLB-DP] SendRankTableAlgorithmInfo tlvData.len:[%u] success.", datlen);

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_ADJ;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        nslbdp_handle_, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));

    HCCL_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_TBL_ADJ(1003) to hccp. ret(%d)\n", ret);
    return HCCL_SUCCESS;
}

/* 遍历表3 执行send 流程 */
HcclResult hcclNslbDp::SendAlgorithmInfoTable()
{
    u32 size = hcclNslbDpAlgorithmInfo_.size();
    HCCL_INFO("[NSLB-DP] ndlbdp entry SendAlgorithmInfoTable size=[%u].", size);

    for (size_t i = 0; i < hcclNslbDpAlgorithmInfo_.size(); i++) {
        if (hcclNslbDpAlgorithmInfo_[i].sedFlag == 1) {
            continue;
        }
        NslbDpAlgorithmTlv tab_f = {};
        tab_f.taskId = hcclNslbDpAlgorithmInfo_[i].taskId;
        s32 sRet = memcpy_s(
            tab_f.commDesc, sizeof(tab_f.commDesc), hcclNslbDpAlgorithmInfo_[i].commDesc, COMM_DESC_MAX_LENGTH);
        sRet = memcpy_s(
            tab_f.commMd5Sum, sizeof(tab_f.commMd5Sum), hcclNslbDpAlgorithmInfo_[i].commMd5Sum,
            sizeof(hcclNslbDpAlgorithmInfo_[i].commMd5Sum));
        if (sRet != EOK) {
            HCCL_INFO("memcpy_s commDesc fail");
        }
        tab_f.srcLocalRankId = hcclNslbDpAlgorithmInfo_[i].srcLocalRankId;
        tab_f.rootRank = hcclNslbDpAlgorithmInfo_[i].rootRank;
        tab_f.oper = hcclNslbDpAlgorithmInfo_[i].oper;
        tab_f.algorithm = hcclNslbDpAlgorithmInfo_[i].algorithm;

        tab_f.rev = 0;

        tab_f.dstRankNum = hcclNslbDpAlgorithmInfo_[i].dstRankNum;
        tab_f.revsecond = 0;
        tab_f.AdjInfo = hcclNslbDpAlgorithmInfo_[i].AdjInfo;

        SendRankTableAlgorithmInfo(tab_f);
        HCCL_INFO("[NSLB-DP] try to sen AlgorithmInfoTable times:[%u].", i);

        hcclNslbDpAlgorithmInfo_[i].sedFlag = 1;
    }
    HCCL_DEBUG("[NSLB-DP] entry SendAlgorithmInfoTable end.");

    return HCCL_SUCCESS;
}

/* 序列化表4 */
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableGlobalRankInfo(NslbDpGlobalRankInfo& info)
{
    HCCL_DEBUG("[NSLB-DP] entry serializeTLV_TableGlobalRankInfo.");
    std::vector<uint8_t> tlvData;
    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), info.commDesc, info.commDesc + COMM_DESC_MAX_LENGTH);

    // 处理 commInitTime
    uint64_t netCommInitTime = htobe64(info.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime,
        sizeof(netCommInitTime));

    // 处理 packetId
    uint16_t netPacketId = htons(info.packetId);
    tlvData.resize(tlvData.size() + sizeof(netPacketId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netPacketId), sizeof(netPacketId), &netPacketId, sizeof(netPacketId));

    // 处理 rev
    uint16_t netRev = htons(info.rev);
    tlvData.resize(tlvData.size() + sizeof(netRev));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev), sizeof(netRev), &netRev, sizeof(netRev));

    // 处理 packetNum
    uint16_t netPacketNum = htons(info.packetNum);
    tlvData.resize(tlvData.size() + sizeof(netPacketNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netPacketNum), sizeof(netPacketNum), &netPacketNum,
        sizeof(netPacketNum));
    HCCL_INFO("[NSLB-DP] hcclNslbDpGlobalRankVal_ info.packetNum:[%u]", netPacketNum);

    // 处理 rev2
    uint16_t netRev2 = htons(info.rev2);
    tlvData.resize(tlvData.size() + sizeof(netRev2));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev2), sizeof(netRev2), &netRev2, sizeof(netRev2));

    // 处理 sendRankInfo
    for (const auto& rankInfo : info.sendRankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp,
            sizeof(netDeviceIp));
    }

    // 处理 commMd5Sum
    tlvData.insert(tlvData.end(), info.commMd5Sum, info.commMd5Sum + sizeof(info.commMd5Sum));

    // 处理 rankTotalNum
    uint32_t netRankTotalNum = htonl(info.rankTotalNum);
    tlvData.resize(tlvData.size() + sizeof(netRankTotalNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRankTotalNum), sizeof(netRankTotalNum), &netRankTotalNum,
        sizeof(netRankTotalNum));

    // 处理 rankNum
    uint16_t netRankNum = htons(info.rankNum);
    tlvData.resize(tlvData.size() + sizeof(netRankNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRankNum), sizeof(netRankNum), &netRankNum, sizeof(netRankNum));

    // 处理 rev3
    uint16_t netRev3 = htons(info.rev3);
    tlvData.resize(tlvData.size() + sizeof(netRev3));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev3), sizeof(netRev3), &netRev3, sizeof(netRev3));

    // 处理 rankInfo
    for (const auto& rankInfo : info.rankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        uint32_t netServerIp = htonl(rankInfo.serverIp);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp,
            sizeof(netDeviceIp));
        tlvData.resize(tlvData.size() + sizeof(netServerIp));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netServerIp), sizeof(netServerIp), &netServerIp,
            sizeof(netServerIp));
    }
    return tlvData;
}

HcclResult hcclNslbDp::SendRankTableGlobalRank(NslbDpGlobalRankInfo& tab_f)
{
    if (GetInitNetCoFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableGlobalRankInfo(tab_f);
    u32 datlen = tlvData.size();
    HCCL_INFO("[NSLB-DP] SendRankTableGlobalRank tlvData.len:[%u] success.", datlen);

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_RANK;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        nslbdp_handle_, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));

    HCCL_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_TBL_RANK(1004) to hccp. ret(%d)\n", ret);

    return HCCL_SUCCESS;
}

void hcclNslbDp::fullCommonGlobalRankInfo(NslbDpGlobalRankInfo& tab_f, NslbDpGlobalRankVal& cominfo)
{
    tab_f.rev = 0;
    tab_f.rev2 = 0;
    tab_f.rev3 = 0;
    tab_f.taskId = cominfo.taskId;
    s32 sRet = memcpy_s(tab_f.commDesc, sizeof(tab_f.commDesc), cominfo.commDesc, sizeof(cominfo.commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commDesc fail");
    }
    tab_f.commInitTime = cominfo.commInitTime;
    sRet = memcpy_s(tab_f.commMd5Sum, sizeof(tab_f.commMd5Sum), cominfo.commMd5Sum, sizeof(cominfo.commMd5Sum));
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commMD5 fail");
    }

    u32 sendCnt = std::min(static_cast<u32>(tab_f.packetNum), static_cast<u32>(cominfo.rankInfo.size()));
    for (u32 dip = 0; dip < sendCnt; dip++) {
        tab_f.sendRankInfo[dip].deviceIp = cominfo.rankInfo[dip].deviceIp;
    }
    return;
}

/* 根据表4 中的rank数量进行分片处理 */
HcclResult hcclNslbDp::SendTableGlobalRankProc(uint32_t rank, uint32_t packetNum, NslbDpGlobalRankVal& cominfo)
{
    HCCL_INFO("[NSLB-DP] entry SendTableGlobalRankProc packetNum:[%u]", packetNum);
    if (packetNum - 1 < rank) {
        HCCL_INFO("[NSLB-DP] rank [%u] no need send TableFir(packetNum[%u]).", rank, packetNum);
        return HCCL_SUCCESS;
    }
    u32 packetIndex = rank;
    NslbDpGlobalRankInfo tab_f = {};
    tab_f.packetId = packetIndex;
    tab_f.packetNum = packetNum;
    fullCommonGlobalRankInfo(tab_f, cominfo);

    uint32_t start = packetIndex * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    uint32_t end = (packetIndex + 1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
    uint32_t totalSize = static_cast<uint32_t>(cominfo.rankInfo.size());
    end = std::min(end, totalSize);
    if (start >= end) {
        HCCL_INFO("[NSLB-DP] GlobalRank RankInfo empty range[%u-%u], totalSize[%u]", start, end, totalSize);
    } else {
        tab_f.rankInfo.reserve(end - start);
        for (uint32_t j = start; j < end; ++j) {
            tab_f.rankInfo.push_back(cominfo.rankInfo[j]);
        }
    }
    tab_f.rankTotalNum = cominfo.rankTotalNum;
    tab_f.rankNum = static_cast<u16>(tab_f.rankInfo.size());
    SendRankTableGlobalRank(tab_f);
    return HCCL_SUCCESS;
}

/* 根据rankNum的总数做分片发送流程 */
HcclResult hcclNslbDp::SendGlobalRankTable(uint32_t rank)
{
    HCCL_DEBUG("[NSLB-DP] try to send GlobalRankTable.");
    u32 rankTotalNum = hcclNslbDpGlobalRankVal_.rankTotalNum;
    if (rankTotalNum > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
        return HCCL_SUCCESS;
    }
    u32 packetNum = CalcPacketNum(rankTotalNum);
    SendTableGlobalRankProc(rank, packetNum, hcclNslbDpGlobalRankVal_);

    return HCCL_SUCCESS;
}

/* 序列化表5 */
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableGlobalDisRankVal(NslbDpGlobalDisRankVal& info)
{
    HCCL_DEBUG("[NSLB-DP] entry serializeTLV_TableGlobalDisRankVal.");
    std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 npuIp
    uint32_t netNpuIp = htonl(info.npuIp);
    tlvData.resize(tlvData.size() + sizeof(netNpuIp));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netNpuIp), sizeof(netNpuIp), &netNpuIp, sizeof(netNpuIp));

    // 处理 serverIp
    uint32_t netServerIp = htonl(info.serverIp);
    tlvData.resize(tlvData.size() + sizeof(netServerIp));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netServerIp), sizeof(netServerIp), &netServerIp, sizeof(netServerIp));

    // 处理 nodeId
    uint32_t netNodeId = htonl(info.nodeId);
    tlvData.resize(tlvData.size() + sizeof(netNodeId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netNodeId), sizeof(netNodeId), &netNodeId, sizeof(netNodeId));

    // 处理 localRankNum
    tlvData.push_back(info.localRankNum);

    // 处理 rev
    tlvData.insert(tlvData.end(), info.rev, info.rev + sizeof(info.rev));

    // 处理 rankTotalNum
    uint32_t netRankTotalNum = htonl(info.rankTotalNum);
    tlvData.resize(tlvData.size() + sizeof(netRankTotalNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRankTotalNum), sizeof(netRankTotalNum), &netRankTotalNum,
        sizeof(netRankTotalNum));

    return tlvData;
}

/* 发送GlobalDisRank表信息 */
HcclResult hcclNslbDp::SendRankTableGlobalDisRankVal(NslbDpGlobalDisRankVal& tab_f)
{
    if (GetInitNetCoFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableGlobalDisRankVal(tab_f);
    u32 datlen = tlvData.size();
    HCCL_INFO("[NSLB-DP] SendRankTableGlobalDisRankVal tlvData.len:[%u] success.", datlen);

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_RANK_DIST;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        nslbdp_handle_, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));

    HCCL_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_TBL_RANK_DIST(1005) to hccp. ret(%d)\n", ret);
    return HCCL_SUCCESS;
}

/* 遍历表5 执行send流程 */
HcclResult hcclNslbDp::SendGlobalDisRankTable()
{
    HCCL_DEBUG("[NSLB-DP] ndlbdp entry serializeTLV_TableGlobalDisRankVal.");
    NslbDpGlobalDisRankVal tab_f = {};
    tab_f.taskId = hcclNslbDpGlobalDisRankVal_.taskId;
    tab_f.npuIp = hcclNslbDpGlobalDisRankVal_.npuIp;
    tab_f.serverIp = hcclNslbDpGlobalDisRankVal_.serverIp;
    tab_f.nodeId = hcclNslbDpGlobalDisRankVal_.nodeId;
    tab_f.localRankNum = hcclNslbDpGlobalDisRankVal_.localRankNum;
    s32 sRet = memcpy_s(
        tab_f.rev, sizeof(tab_f.rev), hcclNslbDpGlobalDisRankVal_.rev, sizeof(hcclNslbDpGlobalDisRankVal_.rev));
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s rev info fail");
    }
    tab_f.rankTotalNum = hcclNslbDpGlobalDisRankVal_.rankTotalNum;

    SendRankTableGlobalDisRankVal(tab_f);
    return HCCL_SUCCESS;
}

/* 序列化表6 */
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableRootRank(NslbDpRootRank& config)
{
    HCCL_DEBUG("[NSLB-DP] ndlbdp entry serializeTLV_TableRootRank.");
    std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(config.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), config.commDesc, config.commDesc + COMM_DESC_MAX_LENGTH);

    // 处理 commInitTime
    uint64_t netCommInitTime = htobe64(config.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime,
        sizeof(netCommInitTime));

    // 处理 oper
    tlvData.push_back(config.oper);

    // 处理 algorithm
    tlvData.push_back(config.algorithm);

    // 处理 revfir
    uint16_t netRevfir = htons(config.revfir);
    tlvData.resize(tlvData.size() + sizeof(netRevfir));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRevfir), sizeof(netRevfir), &netRevfir, sizeof(netRevfir));

    // 处理 rootRankNum
    uint16_t netRootRankNum = htons(config.rootRankNum);
    tlvData.resize(tlvData.size() + sizeof(netRootRankNum));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRootRankNum), sizeof(netRootRankNum), &netRootRankNum,
        sizeof(netRootRankNum));

    // 处理 revsec
    uint16_t netRevsec = htons(config.revsec);
    tlvData.resize(tlvData.size() + sizeof(netRevsec));
    (void)memcpy_s(
        tlvData.data() + tlvData.size() - sizeof(netRevsec), sizeof(netRevsec), &netRevsec, sizeof(netRevsec));

    // 处理 rankId
    for (const auto& id : config.rankId) {
        uint32_t netRankId = htonl(id.rankID);
        tlvData.resize(tlvData.size() + sizeof(netRankId));
        (void)memcpy_s(
            tlvData.data() + tlvData.size() - sizeof(netRankId), sizeof(netRankId), &netRankId, sizeof(netRankId));
    }
    return tlvData;
}

/* 发送Rank表到netco */
HcclResult hcclNslbDp::SendRankTableRootRank(NslbDpRootRank& tab_f)
{
    if (GetInitNetCoFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableRootRank(tab_f);

    u32 datlen = tlvData.size();
    HCCL_DEBUG("[NSLB-DP] SendRankTableRootRank tlvData.len:[%u] success.", datlen);

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_ROOT_RANK;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        nslbdp_handle_, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));

    HCCL_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_TBL_ROOT_RANK(1006) to hccp. ret(%d)\n", ret);
    return HCCL_SUCCESS;
}

/* 遍历表6 执行send流程 */
HcclResult hcclNslbDp::SendRootRankTable()
{
    HCCL_DEBUG("[NSLB-DP] entry SendRootRankTable.");
    NslbDpRootRank tab_f = {};
    tab_f.taskId = hcclNslbDpRootRankVal_.taskId;
    s32 sRet = memcpy_s(
        tab_f.commDesc, sizeof(tab_f.commDesc), hcclNslbDpRootRankVal_.commDesc,
        sizeof(hcclNslbDpRootRankVal_.commDesc));
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s commDesc info fail");
    }
    tab_f.commInitTime = hcclNslbDpRootRankVal_.commInitTime;
    tab_f.oper = hcclNslbDpRootRankVal_.oper;
    tab_f.algorithm = hcclNslbDpRootRankVal_.algorithm;
    tab_f.revfir = hcclNslbDpRootRankVal_.revfir;
    tab_f.rootRankNum = hcclNslbDpRootRankVal_.rootRankNum;
    tab_f.revsec = hcclNslbDpRootRankVal_.revsec;
    tab_f.rankId = hcclNslbDpRootRankVal_.rankId;

    for (const auto& rank : hcclNslbDpRootRankVal_.rankId) {
        tab_f.rankId.push_back(rank);
    }

    SendRankTableRootRank(tab_f);

    return HCCL_SUCCESS;
}

/* 初始化NetCo通道 */
HcclResult hcclNslbDp::InitNetCo()
{
    if (hcclH2dTlv::GetInstance().GetH2dTlvInitFlag() != true) {
        HCCL_INFO("Check GetH2dTlvInitFlag is not success");
        return HCCL_SUCCESS;
    }
    /* 避免二次初始化 */
    if (GetInitNetCoFlag() == true) {
        HCCL_INFO("Get getHccpInitFlag is true");
        return HCCL_SUCCESS;
    }
    if (hcclH2dTlv::GetInstance().GetH2dTlvBufferSize() == NSLBDP_ILLEGAL_TLVBUFFERSIZE) {
        /* 异常场景处理 */
        HCCL_INFO("Check H2dTlvBufferSize equal 0.");
        ClearInitNetCoFlag();
        return HCCL_E_NOT_SUPPORT;
    }
    if (hcclH2dTlv::GetInstance().GetH2dTlvHandle() == nullptr) {
        HCCL_ERROR("Check InitNetCo handle is null.");
        ClearInitNetCoFlag();
        return HCCL_E_NOT_SUPPORT;
    }
    u32 nslbBuffersize = hcclH2dTlv::GetInstance().GetH2dTlvBufferSize();
    void* tlvHandle = hcclH2dTlv::GetInstance().GetH2dTlvHandle();
    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_INIT_NETCO;
    sendMsg.length = NSLBDP_ILLEGAL_MSGLENGTH;

    /* init/deinit 场景date信息默认填成数字0，hccp不关注此字段，但数据不能为NULL */
    std::vector<uint8_t> tlvData;
    tlvData.push_back(0);
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        tlvHandle, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));
    HCCL_RUN_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_INIT_NETCO(9001) to hccp. ret(%d)\n", ret);
    if (ret != 0) {
        return HCCL_E_NOT_SUPPORT;
    }
    /* 数据转存 */
    SetH2DTlvInitInfo(nslbBuffersize, tlvHandle);
    HCCL_DEBUG("Entry InitNetCo end");
    return HCCL_SUCCESS;
}

/* 去初始化NetCo通道 */
void hcclNslbDp::DeinitNetCo()
{
    if (GetInitNetCoFlag() == false) {
        return;
    }
    if (hcclH2dTlv::GetInstance().GetH2dTlvBufferSize() == NSLBDP_ILLEGAL_TLVBUFFERSIZE) {
        return;
    }
    if (hcclH2dTlv::GetInstance().GetH2dTlvHandle() == nullptr) {
        return;
    }
    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_DEINIT_NETCO;
    sendMsg.length = NSLBDP_ILLEGAL_MSGLENGTH;
    /* init/deinit 场景date信息默认填成数字0，hccp不关注此字段，但数据不能为NULL */
    std::vector<uint8_t> tlvData;
    tlvData.push_back(0);
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(
        nslbdp_handle_, MODULE_TYPE_NSLB, reinterpret_cast<TlvMsg*>(&sendMsg), reinterpret_cast<TlvMsg*>(&recvMsg));
    HCCL_RUN_INFO("[NSLBDP-SENDTABLE] hccl send table NSLBDP_TYPE_DEINIT_NETCO(9002) to hccp. ret(%d)\n", ret);
    ClearInitNetCoFlag();
    return;
}

} // namespace hccl
