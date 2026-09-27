/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_config_legacy.h"
#include <fstream>
#include <algorithm>
#include <array>
#include "log.h"

namespace Hccl {

// EnvHostNicConfig

void EnvHostNicConfig::Parse()
{
    hcclIfIp.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_IF_IP set by %s to [%s]", hcclIfIp.GetSource(), GetControlIfIp().Describe().c_str());

    hcclIfBasePort.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_IF_BASE_PORT set by %s to [%u]", hcclIfBasePort.GetSource(), GetIfBasePort());

    hcclSocketIfName.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_SOCKET_IFNAME set by %s to [%s]", hcclSocketIfName.GetSource(),
        GetSocketIfName().configIfNameStr.c_str());

    whitelistDisable.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_WHITELIST_DISABLE set by %s to [%d]", whitelistDisable.GetSource(), whitelistDisable.Get());

    if (!whitelistDisable.Get()) {
        hcclWhiteListFile.Parse();
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_WHITELIST_FILE set by %s to [%s]", hcclWhiteListFile.GetSource(),
            GetWhiteListFile().c_str());
    }

    hcclHostSocketPortRange.Parse();
    std::ostringstream hosrPortRangeOss;
    for (auto range : GetHostSocketPortRange()) {
        hosrPortRangeOss << "[" << range.min << ", " << range.max << "]";
    }
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_HOST_SOCKET_PORT_RANGE set by %s to %s", hcclHostSocketPortRange.GetSource(),
        hosrPortRangeOss.str().c_str());

    hcclDeviceSocketPortRange.Parse();
    std::ostringstream devicePortRangeOss;
    for (auto range : GetDeviceSocketPortRange()) {
        devicePortRangeOss << "[" << range.min << ", " << range.max << "]";
    }
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_NPU_SOCKET_PORT_RANGE set by %s to %s", hcclDeviceSocketPortRange.GetSource(),
        devicePortRangeOss.str().c_str());
}

const IpAddress& EnvHostNicConfig::GetControlIfIp() const { return hcclIfIp.Get(); }

u32 EnvHostNicConfig::GetIfBasePort() const { return hcclIfBasePort.Get(); }

const SocketIfName& EnvHostNicConfig::GetSocketIfName() const { return hcclSocketIfName.Get(); }

bool EnvHostNicConfig::GetWhitelistDisable() const { return whitelistDisable.Get(); }

const std::string& EnvHostNicConfig::GetWhiteListFile() const { return hcclWhiteListFile.Get(); }

const std::vector<SocketPortRange>& EnvHostNicConfig::GetHostSocketPortRange() const
{
    return hcclHostSocketPortRange.Get();
}

const std::vector<SocketPortRange>& EnvHostNicConfig::GetDeviceSocketPortRange() const
{
    return hcclDeviceSocketPortRange.Get();
}

// EnvSocketConfig

void EnvSocketConfig::Parse()
{
    hcclSocketFamily.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_SOCKET_FAMILY set by %s to [%d]", hcclSocketFamily.GetSource(), GetSocketFamily());

    linkTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_CONNECT_TIMEOUT set by %s to [%d]s", linkTimeOut.GetSource(), GetLinkTimeOut());
}

s32 EnvSocketConfig::GetSocketFamily() const { return hcclSocketFamily.Get(); }

s32 EnvSocketConfig::GetLinkTimeOut() const { return linkTimeOut.Get(); }

// EnvRtsConfig

void EnvRtsConfig::Parse()
{
    execTimeOut.Parse();
    aivExecTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_EXEC_TIMEOUT set by %s to [%u]s", execTimeOut.GetSource(), GetExecTimeOut());
}

u32 EnvRtsConfig::GetExecTimeOut() const { return execTimeOut.Get(); }

double EnvRtsConfig::GetAivExecTimeOut() const { return aivExecTimeOut.Get(); }

// EnvRdmaConfig

void EnvRdmaConfig::Parse()
{
    rdmaTrafficClass.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TC set by %s to [%u]", rdmaTrafficClass.GetSource(), GetRdmaTrafficClass());

    rdmaServerLevel.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_SL set by %s to [%u]", rdmaServerLevel.GetSource(), GetRdmaServerLevel());

    rdmaTimeOut.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TIMEOUT set by %s to [%u]", rdmaTimeOut.GetSource(), GetRdmaTimeOut());

    rdmaRetryCnt.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_RETRY_CNT set by %s to [%u]", rdmaRetryCnt.GetSource(), GetRdmaRetryCnt());

    queueNum.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_QPS_PER_CONNECTION set by %s to [%u]", queueNum.GetSource(), GetRdmaQueueNum());

    multiQpThreshold.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_MULTI_QP_THRESHOLD set by %s to [%u]B", multiQpThreshold.GetSource(),
        GetRdmaMultiQpThreshold());

    hostRdmaUdpPortsList.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_HOST_RDMA_UDP_PORTS_LIST set by %s, device config count[%zu]",
        hostRdmaUdpPortsList.GetSource(), GetHostRdmaUdpPortsList().GetDeviceConfigCount());

    qpPortConfigPath.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_RDMA_QP_PORT_CONFIG_PATH set by %s to [%s]", qpPortConfigPath.GetSource(),
        qpPortConfigPath.Get().c_str());
    ParseMultiQpSrcPortConfig();
}

HcclResult EnvRdmaConfig::OpenMultiQpConfigFile(std::ifstream& inFile)
{
    std::string fileStr = qpPortConfigPath.Get() + "/MultiQpSrcPort.cfg";
    std::array<char, PATH_MAX> realFile{};
    if (realpath(fileStr.c_str(), realFile.data()) == nullptr) {
        HCCL_ERROR("[EnvRdmaConfig][OpenMultiQpConfigFile] config file path[%s] is invalid.", fileStr.c_str());
        return HCCL_E_PARA;
    }

    inFile.open(realFile.data(), std::ifstream::in);
    if (!inFile) {
        HCCL_ERROR("[EnvRdmaConfig][OpenMultiQpConfigFile] open config file[%s] failed.", realFile.data());
        return HCCL_E_PARA;
    }
    HCCL_INFO("[EnvRdmaConfig][OpenMultiQpConfigFile] open config file[%s] success.", realFile.data());
    return HCCL_SUCCESS;
}

HcclResult EnvRdmaConfig::ParseConfigContent(std::ifstream& inFile, MultiQpSrcPortConfig& config)
{
    config.configDirPath = qpPortConfigPath.Get();
    u32 lineCnt = 1;
    std::string line;
    while (std::getline(inFile, line)) {
        std::string lineAvator = line;
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
        auto hashPos = line.find('#');
        std::string lineInfo = (hashPos != std::string::npos) ? line.substr(0, hashPos) : line;
        if (lineInfo.empty()) {
            lineCnt++;
            continue;
        }

        std::string ipPairKey, portPart;
        HcclResult ret = ParseLineToIpPairAndPortPart(lineInfo, lineCnt, lineAvator, ipPairKey, portPart);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        std::vector<std::uint16_t> ports;
        ret = ParseSrcPortsFromPortPart(portPart, lineCnt, lineAvator, ports);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        CHK_PRT_RET(
            config.ipPairToPorts.find(ipPairKey) != config.ipPairToPorts.end(),
            HCCL_ERROR(
                "[MulQpInfo][ParseSrcPortsFromString][line: %u] ip pair[%s] already exists.[%s]", lineCnt,
                ipPairKey.c_str(), lineAvator.c_str()),
            HCCL_E_PARA);
        config.ipPairToPorts[ipPairKey] = ports;
        if (lineCnt >= MultiQpSrcPortConfig::CONFIG_FILE_LINE_MAX) {
            HCCL_RUN_INFO("config file is too large, stop parsing at line %u.", lineCnt);
            break;
        }
        lineCnt++;
    }
    return HCCL_SUCCESS;
}

void EnvRdmaConfig::ParseMultiQpSrcPortConfig()
{
    if (!qpPortConfigPath.IsSetByEnvironment() || qpPortConfigPath.Get().empty()) {
        return;
    }
    try {
        std::ifstream inFile;
        CHK_PRT_THROW(
            OpenMultiQpConfigFile(inFile) != HCCL_SUCCESS,
            HCCL_ERROR("[EnvRdmaConfig][Parse] open multi qp src port config file failed."), InvalidParamsException,
            "open config file fail.");

        MultiQpSrcPortConfig config;
        CHK_PRT_THROW(
            ParseConfigContent(inFile, config) != HCCL_SUCCESS,
            HCCL_ERROR("[EnvRdmaConfig][Parse] parse multi qp src port config content failed."), InvalidParamsException,
            "parse config content fail.");

        inFile.close();
        multiQpSrcPortConfig_ = config;
        LogMultiQpSrcPortConfig();
    } catch (const HcclException& e) {
        HCCL_ERROR("[EnvRdmaConfig][Parse] LoadMultiQpSrcPortConfig failed: %s", e.what());
    } catch (const std::exception& e) {
        HCCL_ERROR("[EnvRdmaConfig][Parse] LoadMultiQpSrcPortConfig failed: %s", e.what());
    }
}

HcclResult EnvRdmaConfig::ParseLineToIpPairAndPortPart(
    const std::string& lineInfo, u32 lineCnt, const std::string& lineAvator, std::string& ipPairKey,
    std::string& portPart)
{
    auto eqPos = lineInfo.find('=');
    CHK_PRT_RET(
        eqPos == std::string::npos || eqPos == 0 || eqPos == lineInfo.size() - 1,
        HCCL_ERROR(
            "[MulQpInfo][ParseSrcPortsFromString][line: %u]Expected format: "
            "'srcIP,dstIP=srcPort0,srcPort1,...'.[%s]",
            lineCnt, lineAvator.c_str()),
        HCCL_E_PARA);
    CHK_PRT_RET(
        lineInfo.find('=', eqPos + 1) != std::string::npos,
        HCCL_ERROR(
            "[MulQpInfo][ParseSrcPortsFromString][line: %u]Expected format: "
            "'srcIP,dstIP=srcPort0,srcPort1,...'.[%s]",
            lineCnt, lineAvator.c_str()),
        HCCL_E_PARA);
    std::string ipPart = lineInfo.substr(0, eqPos);
    portPart = lineInfo.substr(eqPos + 1);

    auto commaPos = ipPart.find(',');
    CHK_PRT_RET(
        commaPos == std::string::npos,
        HCCL_ERROR(
            "[MulQpInfo][ParseSrcPortsFromString][line: %u] IP pair format error, "
            "expected 'srcIP,dstIP'.[%s]",
            lineCnt, lineAvator.c_str()),
        HCCL_E_PARA);
    CHK_PRT_RET(
        ipPart.find(',', commaPos + 1) != std::string::npos,
        HCCL_ERROR(
            "[MulQpInfo][ParseSrcPortsFromString][line: %u] IP pair format error, "
            "expected 'srcIP,dstIP'.[%s]",
            lineCnt, lineAvator.c_str()),
        HCCL_E_PARA);

    IpAddress srcIpAddr, dstIpAddr;
    try {
        srcIpAddr = IpAddress(ipPart.substr(0, commaPos));
        dstIpAddr = IpAddress(ipPart.substr(commaPos + 1));
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(
            "[MulQpInfo][ParseSrcPortsFromString][line: %u] IP format error: %s.[%s]", lineCnt, e.what(),
            lineAvator.c_str());
        return HCCL_E_PARA;
    }

    ipPairKey = srcIpAddr.GetIpStr() + "," + dstIpAddr.GetIpStr();
    return HCCL_SUCCESS;
}

HcclResult EnvRdmaConfig::ParseSrcPortsFromPortPart(
    const std::string& portPart, u32 lineCnt, const std::string& lineAvator, std::vector<std::uint16_t>& ports)
{
    std::size_t start = 0;
    while (true) {
        auto p = portPart.find(',', start);
        std::string token = (p != std::string::npos) ? portPart.substr(start, p - start) : portPart.substr(start);

        CHK_PRT_RET(
            token.empty(),
            HCCL_ERROR(
                "[MulQpInfo][ParseSrcPortsFromString][line: %u]src port[%s] is empty.[%s]", lineCnt, token.c_str(),
                lineAvator.c_str()),
            HCCL_E_PARA);

        unsigned long val = 0;
        try {
            std::size_t parsePos = 0;
            val = std::stoul(token, &parsePos, 0);
            if (parsePos != token.size()) {
                HCCL_ERROR(
                    "[MulQpInfo][ParseSrcPortsFromString][line: %u] port[%s] is not a valid integer.[%s]", lineCnt,
                    token.c_str(), lineAvator.c_str());
                return HCCL_E_PARA;
            }
        } catch (...) {
            HCCL_ERROR(
                "[MulQpInfo][ParseSrcPortsFromString][line: %u] port[%s] is not a valid integer.[%s]", lineCnt,
                token.c_str(), lineAvator.c_str());
            return HCCL_E_PARA;
        }

        CHK_PRT_RET(
            val == 0 || val > MultiQpSrcPortConfig::CONFIG_SRC_PORT_ID_MAX,
            HCCL_ERROR(
                "[MulQpInfo][ParseSrcPortsFromString][line: %u]src port[%s] "
                "should be within the range of[1, %u].[%s]",
                lineCnt, token.c_str(), MultiQpSrcPortConfig::CONFIG_SRC_PORT_ID_MAX, lineAvator.c_str()),
            HCCL_E_PARA);

        ports.emplace_back(static_cast<std::uint16_t>(val));
        if (p == std::string::npos)
            break;
        start = p + 1;
    }

    CHK_PRT_RET(
        ports.size() > MultiQpSrcPortConfig::CONFIG_SRC_PORT_NUM_MAX || ports.empty(),
        HCCL_ERROR(
            "[MulQpInfo][ParseSrcPortsFromString][line: %u]config ports num[%zu] more than the "
            "threshold[%u].[%s]",
            lineCnt, ports.size(), MultiQpSrcPortConfig::CONFIG_SRC_PORT_NUM_MAX, lineAvator.c_str()),
        HCCL_E_PARA);

    return HCCL_SUCCESS;
}

void EnvRdmaConfig::LogMultiQpSrcPortConfig() const
{
    for (const auto& entry : multiQpSrcPortConfig_.ipPairToPorts) {
        std::string portsStr;
        for (size_t j = 0; j < entry.second.size(); j++) {
            if (j > 0)
                portsStr += ",";
            portsStr += std::to_string(entry.second[j]);
        }
        HCCL_RUN_INFO("[HCCL_ENV] MultiQpSrcPort: ipPair[%s] -> ports[%s]", entry.first.c_str(), portsStr.c_str());
    }
}

u32 EnvRdmaConfig::GetRdmaTrafficClass() const { return rdmaTrafficClass.Get(); }

u32 EnvRdmaConfig::GetRdmaServerLevel() const { return rdmaServerLevel.Get(); }

u32 EnvRdmaConfig::GetRdmaTimeOut() const { return rdmaTimeOut.Get(); }

u32 EnvRdmaConfig::GetRdmaRetryCnt() const { return rdmaRetryCnt.Get(); }

u32 EnvRdmaConfig::GetRdmaQueueNum() const { return queueNum.Get(); }

u32 EnvRdmaConfig::GetRdmaMultiQpThreshold() const { return multiQpThreshold.Get(); }

const HostRdmaUdpPortsList& EnvRdmaConfig::GetHostRdmaUdpPortsList() const { return hostRdmaUdpPortsList.Get(); }

const MultiQpSrcPortConfig& EnvRdmaConfig::GetMultiQpSrcPortConfig() const { return multiQpSrcPortConfig_; }

// EnvAlgoConfig

void EnvAlgoConfig::Parse()
{
    // Hcomm中不再解析HCCL_ALGO配置，因为没有实际使用
    primQueueGenName.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] PRIM_QUEUE_GEN_NAME set by %s to [%s]", primQueueGenName.GetSource(),
        GetPrimQueueGenName().c_str());

    bufferSize.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_BUFFSIZE set by %s to [%llu]B", bufferSize.GetSource(), GetBuffSize());

    hcclAccelerator_.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_OP_EXPANSION_MODE set by %s to [%s]", hcclAccelerator_.GetSource(),
        GetHcclAccelerator().Describe().c_str());
}

const std::string& EnvAlgoConfig::GetPrimQueueGenName() const { return primQueueGenName.Get(); }

const std::map<OpType, std::vector<HcclAlgoType>> EnvAlgoConfig::GetAlgoConfig() const
{
    // hcomm不支持HCCL_ALGO解析, 返回空
    return {};
}

u64 EnvAlgoConfig::GetBuffSize() const { return bufferSize.Get(); }

HcclAccelerator EnvAlgoConfig::GetHcclAccelerator() const { return hcclAccelerator_.Get(); }

// EnvLogConfig
void EnvLogConfig::Parse()
{
    entryLogEnable.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_ENTRY_LOG_ENABLE set by %s to [%d]", entryLogEnable.GetSource(), GetEntryLogEnable());

    cannVersion.Parse();
    HCCL_RUN_INFO("[HCCL_ENV] LD_LIBRARY_PATH set by %s to [%s]", cannVersion.GetSource(), GetCannVersion().c_str());

    dfsConfig.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_DFS_CONFIG task_exception set by %s to [%d], cluster_heartbeat set by %s to [%d], "
        "rankConsistentState set by %s to [%d]",
        dfsConfig.GetSource(), GetDfsConfig().taskExceptionEnable, dfsConfig.GetSource(),
        GetDfsConfig().clusterHeartBeatEnable, dfsConfig.GetSource(), GetDfsConfig().rankConsistentState);
}

bool EnvLogConfig::GetEntryLogEnable() const { return entryLogEnable.Get(); }

const std::string& EnvLogConfig::GetCannVersion() const { return cannVersion.Get(); }

const DfsConfig& EnvLogConfig::GetDfsConfig() const { return dfsConfig.Get(); }

// EnvDetourConfig

void EnvDetourConfig::Parse()
{
    detourType.Parse();
    HCCL_RUN_INFO(
        "[HCCL_ENV] HCCL_DETOUR set by %s to [%s]", detourType.GetSource(), GetDetourType().Describe().c_str());
}

HcclDetourType EnvDetourConfig::GetDetourType() const { return detourType.Get(); }

} // namespace Hccl
