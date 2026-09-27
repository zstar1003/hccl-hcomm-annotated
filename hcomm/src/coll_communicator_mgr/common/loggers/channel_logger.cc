/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel_logger.h"
#include "comm_addr_logger.h"
#include "endpoint_logger.h"
#include "../endpoint_pairs/channels/channel.h" // ChannelStatus 枚举定义

namespace hcomm {
namespace logger {

    void ChannelLogger::PrintBasicFields(uint32_t idx, const HcclChannelDesc& channelDesc)
    {
        HCCL_INFO(
            "[%s] channelDescs[%u]: remoteRank[%u], channelProtocol[%d], notifyNum[%u], memHandleNum[%u]", __func__,
            idx, channelDesc.remoteRank, channelDesc.channelProtocol, channelDesc.notifyNum, channelDesc.memHandleNum);
    }

    void ChannelLogger::PrintRoceAttributes(uint32_t idx, const HcclChannelDesc& channelDesc)
    {
        if (!IsRoceProtocol(channelDesc)) {
            return;
        }
        std::string roceAttrStr = FormatRoceAttrDetail(channelDesc);
        HCCL_INFO("[%s] channelDescs[%u] roceAttr: %s", __func__, idx, roceAttrStr.c_str());
    }

    void ChannelLogger::PrintDescInfo(uint32_t idx, const HcclChannelDesc& channelDesc)
    {
        // 基本字段
        PrintBasicFields(idx, channelDesc);

        // 本地端点
        EndpointLogger::Print(idx, "localEndpoint", channelDesc.localEndpoint);

        // 远端端点
        EndpointLogger::Print(idx, "remoteEndpoint", channelDesc.remoteEndpoint);

        // ROCE 协议特有属性
        PrintRoceAttributes(idx, channelDesc);

        // UB_MEM 协议特有属性
        PrintUbMemAttributes(idx, channelDesc);
    }

    void ChannelLogger::PrintDescTableHeader()
    {
        HCCL_INFO("   "
                  "_________________________________________________CHANNEL_DESCRIPTOR_INFO____________________________"
                  "_____________________");
        HCCL_INFO("   |  idx  | remoteRank | Proto |  notifyNum  | memHandleNum |              localAddr               "
                  "|             remoteAddr             |           ROCE Attr            |");
        HCCL_INFO("   "
                  "|-------|------------|-------|-------------|--------------|-------------------------------------|---"
                  "----------------------------------|--------------------------------|");
    }

    void ChannelLogger::PrintDescInfoRow(uint32_t idx, const HcclChannelDesc& channelDesc)
    {
        // 使用类内部辅助函数
        std::string localAddr, remoteAddr;
        FormatEndpointAddresses(channelDesc, localAddr, remoteAddr);
        std::string roceAttrStr = FormatRoceAttrCompact(channelDesc);

        // 输出表格行
        HCCL_INFO(
            "   | %5u | %10u | %5d | %11u | %12u | %35s | %35s | %30s |", idx, channelDesc.remoteRank,
            channelDesc.channelProtocol, channelDesc.notifyNum, channelDesc.memHandleNum, localAddr.c_str(),
            remoteAddr.c_str(), roceAttrStr.c_str());
    }

    void ChannelLogger::PrintErrorTableHeader(uint32_t localRank)
    {
        HCCL_ERROR("   "
                   "________________________________________________CHANNEL_CONNECT_ERROR_INFO_________________________"
                   "_______________________");
        HCCL_ERROR("   | comm error, localRank[%u]", localRank);
        HCCL_ERROR(
            "   |  idx  | localRank |              localAddr               | remoteRank |             remoteAddr       "
            "      | chHandle  |            Status            |    TlsStatus    | Proto | elapsed |");
        HCCL_ERROR("   "
                   "|-------|-----------|-------------------------------------|------------|---------------------------"
                   "----------|-----------|----------------------------|-----------------|-------|---------|");
    }

    std::string ChannelStatusUtils::ToString(int32_t status)
    {
        ChannelStatus::Value statusValue = static_cast<ChannelStatus::Value>(status);
        ChannelStatus statusEnum(statusValue);
        return statusEnum.Describe();
    }

    void ChannelLogger::PrintErrorInfo(
        uint32_t idx, uint32_t localRank, const HcclChannelDesc& channelDesc, ChannelHandle channelHandle,
        int32_t status, uint64_t elapsedMs, Hccl::TlsStatus tlsStatus)
    {
        // 复用 FormatEndpointAddresses()
        std::string localAddr, remoteAddr;
        FormatEndpointAddresses(channelDesc, localAddr, remoteAddr);
        std::string statusStr = ChannelStatusUtils::ToString(status);
        std::string tlsStatusStr = "UNKNOWN";
        switch (tlsStatus) {
            case Hccl::TlsStatus::ENABLE:
                tlsStatusStr = "ENABLE";
                break;
            case Hccl::TlsStatus::DISABLE:
                tlsStatusStr = "DISABLE";
                break;
            default:
                tlsStatusStr = "UNKNOWN";
                break;
        }

        HCCL_ERROR(
            "   | %5u | %9u | %35s | %10u | %35s | 0x%08llx | %26s | %15s | %5d | %7llu ms |", idx, localRank,
            localAddr.c_str(), channelDesc.remoteRank, remoteAddr.c_str(),
            static_cast<unsigned long long>(channelHandle), statusStr.c_str(), tlsStatusStr.c_str(),
            channelDesc.channelProtocol, static_cast<unsigned long long>(elapsedMs));
    }

    void ChannelLogger::PrintChannelErrorDetails(
        uint32_t localRank, uint32_t channelNum, const HcclChannelDesc* channelDescs, ChannelHandle* channelHandles,
        int32_t* statusList, int64_t elapsedMs, const Hccl::TlsStatus* tlsStatusList)
    {
        // 打印错误详情表格（只打印异常状态的 Channel）
        PrintErrorTableHeader(localRank);

        for (uint32_t i = 0; i < channelNum; ++i) {
            if (IsAbnormalStatus(statusList[i])) {
                PrintErrorInfo(
                    i, localRank, channelDescs[i], channelHandles[i], statusList[i], elapsedMs, tlsStatusList[i]);
            }
        }

        // 表格外单独打印详细信息（FAILED 或 TIMEOUT 状态）
        for (uint32_t i = 0; i < channelNum; ++i) {
            if (NeedDetailPrint(statusList[i])) {
                PrintDescInfo(i, channelDescs[i]);
            }
        }
    }

    // ========== 私有辅助函数实现 ==========

    bool ChannelLogger::IsRoceProtocol(const HcclChannelDesc& channelDesc)
    {
        return channelDesc.channelProtocol == COMM_PROTOCOL_ROCE;
    }

    std::string ChannelLogger::FormatRoceAttrCompact(const HcclChannelDesc& channelDesc)
    {
        if (!IsRoceProtocol(channelDesc)) {
            return std::string("-");
        }

        char roceAttrStr[128] = {0};
        snprintf_s(
            roceAttrStr, sizeof(roceAttrStr), sizeof(roceAttrStr) - 1, "q:%u r:%u ri:%u tc:%u sl:%u",
            channelDesc.roceAttr.queueNum, channelDesc.roceAttr.retryCnt, channelDesc.roceAttr.retryInterval,
            channelDesc.roceAttr.tc, channelDesc.roceAttr.sl);

        return std::string(roceAttrStr);
    }

    std::string ChannelLogger::FormatRoceAttrDetail(const HcclChannelDesc& channelDesc)
    {
        if (!IsRoceProtocol(channelDesc)) {
            return std::string("");
        }

        char roceAttrStr[256] = {0};
        snprintf_s(
            roceAttrStr, sizeof(roceAttrStr), sizeof(roceAttrStr) - 1,
            "queueNum[%u], retryCnt[%u], retryInterval[%u], tc[%u], sl[%u]", channelDesc.roceAttr.queueNum,
            channelDesc.roceAttr.retryCnt, channelDesc.roceAttr.retryInterval, channelDesc.roceAttr.tc,
            channelDesc.roceAttr.sl);

        return std::string(roceAttrStr);
    }

    bool ChannelLogger::IsAbnormalStatus(int32_t status) { return status != ChannelStatus::READY; }

    bool ChannelLogger::NeedDetailPrint(int32_t status)
    {
        return status == ChannelStatus::FAILED || status == ChannelStatus::SOCKET_TIMEOUT;
    }

    void ChannelLogger::FormatEndpointAddresses(
        const HcclChannelDesc& channelDesc, std::string& outLocalAddr, std::string& outRemoteAddr)
    {
        outLocalAddr = CommAddrLogger::ToString(channelDesc.localEndpoint.commAddr);
        outRemoteAddr = CommAddrLogger::ToString(channelDesc.remoteEndpoint.commAddr);
    }

    void ChannelLogger::PrintUbMemAttributes(uint32_t idx, const HcclChannelDesc& channelDesc)
    {
        if (!IsUbMemProtocol(channelDesc)) {
            return;
        }
        HCCL_INFO("[%s] channelDescs[%u] ubMemAttr: pathMode[%u]", __func__, idx, channelDesc.ubMemAttr.pathMode);
    }

    bool ChannelLogger::IsUbMemProtocol(const HcclChannelDesc& channelDesc)
    {
        return channelDesc.channelProtocol == COMM_PROTOCOL_UB_MEM;
    }

} // namespace logger
} // namespace hcomm
