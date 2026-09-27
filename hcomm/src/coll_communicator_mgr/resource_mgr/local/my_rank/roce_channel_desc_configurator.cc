/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "roce_channel_desc_configurator.h"

#include "adapter_rts_common.h"
#include "env_config/env_config_v2.h"
#include "hcomm_adapter_hccp.h"
#include "orion_adpt_utils.h"

static bool ReadHostConfigValue(RaInfo& info, HccnCfgKey key, std::string& value)
{
    std::vector<char> buffer(hccl::RoceChannelDescConfigurator::HOST_NIC_CONFIG_BUFFER_SIZE, '\0');
    uint32_t valueLen = static_cast<uint32_t>(buffer.size());
    const int ret = RaGetHccnCfg(&info, key, buffer.data(), &valueLen);
    // 配置文件不存在或配置项为空时，RaGetHccnCfg返回成功和空值，不会进入该分支
    if (ret != 0) {
        HCCL_WARNING("[%s] RaGetHccnCfg failed, phyId[%u], key[%d], ret[%d]", __func__, info.phyId, key, ret);
        return false;
    }

    valueLen = std::min<uint32_t>(valueLen, static_cast<uint32_t>(buffer.size()));
    value.assign(buffer.data(), valueLen);
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return true;
}

static bool
ReadHostNicMultiQpConfigItem(uint32_t devicePhyId, HccnCfgKey key, const char* itemName, std::string& itemValue)
{
    RaInfo info{};
    info.mode = NETWORK_PEER_ONLINE;
    info.phyId = devicePhyId;
    std::string mode;
    if (!ReadHostConfigValue(info, HCCN_CFG_UDP_PORT_MODE, mode)) {
        return false;
    }
    if (mode.empty()) {
        HCCL_WARNING("[%s] no host multi-qp config, phyId[%u]", __func__, devicePhyId);
        return false;
    }
    if (mode != "multi_qp") {
        HCCL_WARNING("[%s] invalid udp_port_mode[%s], phyId[%u]", __func__, mode.c_str(), devicePhyId);
        return false;
    }

    if (!ReadHostConfigValue(info, key, itemValue)) {
        return false;
    }
    if (itemValue.empty()) {
        HCCL_WARNING("[%s] incomplete host multi-qp config, phyId[%u] %s[0]", __func__, devicePhyId, itemName);
        return false;
    }
    return true;
}

namespace hccl {

constexpr std::size_t RoceChannelDescConfigurator::HOST_NIC_CONFIG_BUFFER_SIZE;

RoceChannelDescConfigurator::RoceChannelDescConfigurator(uint32_t channelNum) : srcPortBuffers_(channelNum) {}

bool RoceChannelDescConfigurator::ParseStrictDecimal(
    const std::string& value, uint32_t minValue, uint32_t maxValue, uint32_t& parsed)
{
    const auto isDecimalDigit = [](unsigned char character) {
        return std::isdigit(character) != 0;
    };
    const bool hasValidDecimalFormat = !value.empty() && std::all_of(value.begin(), value.end(), isDecimalDigit);
    if (!hasValidDecimalFormat) {
        HCCL_WARNING("[%s] value[%s] is not a valid decimal number", __func__, value.c_str());
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long number = std::strtoul(value.c_str(), &end, 10);
    if (errno == ERANGE || end != value.c_str() + value.size()) {
        HCCL_WARNING("[%s] parse value[%s] failed", __func__, value.c_str());
        return false;
    }
    if (number < minValue || number > maxValue) {
        HCCL_WARNING("[%s] value[%s] is out of range[%u, %u]", __func__, value.c_str(), minValue, maxValue);
        return false;
    }

    parsed = static_cast<uint32_t>(number);
    return true;
}

void RoceChannelDescConfigurator::ReadHostNicMultiQpCount(uint32_t& qpCount)
{
    qpCount = 0;
    s32 deviceLogicId = INVALID_INT;
    uint32_t devicePhyId = INVALID_UINT;
    if ((hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS)
        || (hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devicePhyId, false) != HCCL_SUCCESS)) {
        return;
    }

    std::string count;
    if (!ReadHostNicMultiQpConfigItem(devicePhyId, HCCN_CFG_MULTI_QP_COUNT, "multi_qp_count", count)) {
        return;
    }

    uint32_t parsedQpCount = 0;
    if (!ParseStrictDecimal(count, 1, Hccl::MultiQpSrcPortConfig::CONFIG_SRC_PORT_NUM_MAX, parsedQpCount)) {
        HCCL_WARNING("[%s] invalid multi_qp_count[%s], phyId[%u]", __func__, count.c_str(), devicePhyId);
        return;
    }

    qpCount = parsedQpCount;
    HCCL_INFO("[%s] host config valid, phyId[%u] count[%u]", __func__, devicePhyId, qpCount);
}

void RoceChannelDescConfigurator::ReadHostNicMultiQpUdpPorts(std::vector<uint16_t>& qpUdpPorts)
{
    qpUdpPorts.clear();
    s32 deviceLogicId = INVALID_INT;
    uint32_t devicePhyId = INVALID_UINT;
    if ((hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS)
        || (hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devicePhyId, false) != HCCL_SUCCESS)) {
        return;
    }

    std::string ports;
    if (!ReadHostNicMultiQpConfigItem(devicePhyId, HCCN_CFG_MULTI_QP_UDP_PORTS, "multi_qp_udp_ports", ports)) {
        return;
    }

    std::vector<uint16_t> parsedQpUdpPorts;
    if (!Hccl::ParseHostRdmaUdpPorts(ports, parsedQpUdpPorts)) {
        HCCL_WARNING("[%s] invalid multi_qp_udp_ports[%s], phyId[%u]", __func__, ports.c_str(), devicePhyId);
        return;
    }

    qpUdpPorts = std::move(parsedQpUdpPorts);
    HCCL_INFO("[%s] host config valid, phyId[%u] ports[%zu]", __func__, devicePhyId, qpUdpPorts.size());
}

void RoceChannelDescConfigurator::FillPortsFromHostRdmaUdpPortsList(std::vector<uint16_t>& ports)
{
    const auto& udpPortsList = Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetHostRdmaUdpPortsList();
    if (!udpPortsList.IsAvailable()) {
        HCCL_INFO(
            "[%s] hostRdmaUdpPortsList not available (env HCCL_HOST_RDMA_UDP_PORTS_LIST unset or empty)", __func__);
        return;
    }

    s32 deviceLogicId = INVALID_INT;
    uint32_t devicePhyId = INVALID_UINT;
    if ((hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS)
        || (hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devicePhyId, false) != HCCL_SUCCESS)) {
        HCCL_WARNING("[%s] hrtGetDevice or hrtGetDevicePhyIdByIndex failed, fall back to MultiQpSrcPort.cfg", __func__);
        return;
    }

    const auto portIter = udpPortsList.portsByPhyId.find(devicePhyId);
    if (portIter != udpPortsList.portsByPhyId.end()) {
        ports = portIter->second;
    }
    if (ports.empty()) {
        HCCL_INFO("[%s] no matching HCCL_HOST_RDMA_UDP_PORTS_LIST ports for phyId[%u]", __func__, devicePhyId);
    }
}

HcclResult RoceChannelDescConfigurator::FillPortsFromMultiQpSrcPortConfig(
    const HcclChannelDesc& hcclDesc, std::vector<uint16_t>& ports)
{
    const auto& qpSrcPortConfig = Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetMultiQpSrcPortConfig();
    if (!qpSrcPortConfig.IsAvailable()) {
        HCCL_INFO(
            "[%s] skip: multiQpSrcPortConfig not available (env HCCL_RDMA_QP_PORT_CONFIG_PATH unset or "
            "MultiQpSrcPort.cfg empty)",
            __func__);
        return HCCL_SUCCESS;
    }

    Hccl::IpAddress localIp;
    Hccl::IpAddress remoteIp;
    HcclResult localRet = hcomm::CommAddrToIpAddress(hcclDesc.localEndpoint.commAddr, localIp);
    HcclResult remoteRet = hcomm::CommAddrToIpAddress(hcclDesc.remoteEndpoint.commAddr, remoteIp);
    CHK_PRT_RET(
        localRet != HCCL_SUCCESS || remoteRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CommAddrToIpAddress failed: localRet[%d] remoteRet[%d]", __func__, localRet, remoteRet),
        HCCL_E_INTERNAL);
    ports = Hccl::GetMultiQpSrcPortsByIpPair(qpSrcPortConfig, localIp, remoteIp);
    if (ports.empty()) {
        HCCL_INFO(
            "[%s] skip: no matching ports for localIp[%s] remoteIp[%s]", __func__, localIp.GetIpStr().c_str(),
            remoteIp.GetIpStr().c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult
RoceChannelDescConfigurator::ResolveRoceSrcPorts(const HcclChannelDesc& hcclDesc, std::vector<uint16_t>& ports)
{
    ports.clear();
    ReadHostNicMultiQpUdpPorts(ports);
    if (!ports.empty()) {
        return HCCL_SUCCESS;
    }

    FillPortsFromHostRdmaUdpPortsList(ports);
    if (!ports.empty()) {
        return HCCL_SUCCESS;
    }

    return FillPortsFromMultiQpSrcPortConfig(hcclDesc, ports);
}

HcclResult RoceChannelDescConfigurator::FillRoceSrcPortList(
    const HcclChannelDesc& hcclDesc, uint32_t channelIndex, HcommChannelDesc& hcommDesc)
{
    hcommDesc.roceAttr.srcPortList = nullptr;
    if (hcclDesc.localEndpoint.loc.locType != ENDPOINT_LOC_TYPE_HOST
        || hcommDesc.remoteEndpoint.protocol != COMM_PROTOCOL_ROCE || hcommDesc.exchangeAllMems) {
        HCCL_INFO(
            "[%s] skip: localLocType[%d] protocol[%d] exchangeAllMems[%d]", __func__,
            hcclDesc.localEndpoint.loc.locType, hcommDesc.remoteEndpoint.protocol, hcommDesc.exchangeAllMems);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        channelIndex >= srcPortBuffers_.size(),
        HCCL_ERROR(
            "[%s] channelIndex[%u] is out of range, channelNum[%zu]", __func__, channelIndex, srcPortBuffers_.size()),
        HCCL_E_PARA);

    std::vector<uint16_t> ports;
    CHK_RET(ResolveRoceSrcPorts(hcclDesc, ports));
    if (ports.empty()) {
        return HCCL_SUCCESS;
    }
    const uint32_t queueNum = hcommDesc.roceAttr.queueNum;
    if (ports.size() != static_cast<std::size_t>(queueNum)) {
        HCCL_RUN_WARNING(
            "[%s] UDP source port count[%zu] does not match queueNum[%u], ports will be truncated or reused "
            "cyclically",
            __func__, ports.size(), queueNum);
    }
    auto& srcPortBuffer = srcPortBuffers_[channelIndex];
    srcPortBuffer.resize(queueNum);
    for (uint32_t index = 0; index < queueNum; ++index) {
        srcPortBuffer[index] = ports[index % ports.size()];
    }
    hcommDesc.roceAttr.srcPortList = srcPortBuffer.data();
    HCCL_INFO("[%s] success: queueNum[%u] portCount[%zu]", __func__, queueNum, ports.size());
    return HCCL_SUCCESS;
}

} // namespace hccl
