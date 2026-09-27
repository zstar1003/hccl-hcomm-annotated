/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_BASE_CONFIG_LEGACY_H
#define HCCLV2_BASE_CONFIG_LEGACY_H

#include <string>
#include <vector>
#include <functional>
#include <climits>
#include <regex>
#include "ip_address.h"
#include "dma_mode.h"
#include "env_func.h"
#include "cfg_field.h"

namespace Hccl {

class BaseConfig {
public:
    virtual void Parse() = 0;
};

// Host网卡配置
class EnvHostNicConfig : public BaseConfig {
public:
    void Parse() override;
    const IpAddress& GetControlIfIp() const;
    u32 GetIfBasePort() const;
    const SocketIfName& GetSocketIfName() const;
    bool GetWhitelistDisable() const;
    const std::string& GetWhiteListFile() const;
    const std::vector<SocketPortRange>& GetHostSocketPortRange() const;
    const std::vector<SocketPortRange>& GetDeviceSocketPortRange() const;

private:
    static constexpr u32 HCCL_INVALID_IF_BASE_PORT = 65536;     // HCCL默认无效端口号
    static constexpr u32 HCCL_INVALID_IF_BASE_PORT_MAX = 65520; // HCCL端口号最大值
    static constexpr u32 HCCL_INVALID_IF_BASE_PORT_MIN = 1024;  // HCCL端口号最小值

    CfgField<IpAddress> hcclIfIp{"HCCL_IF_IP", {}, Str2T<IpAddress>};
    CfgField<u32> hcclIfBasePort{
        "HCCL_IF_BASE_PORT", u32(HCCL_INVALID_IF_BASE_PORT), Str2T<u32>,
        CHK_RANGE_CLOSED<u32>(HCCL_INVALID_IF_BASE_PORT_MIN, HCCL_INVALID_IF_BASE_PORT_MAX)};
    CfgField<SocketIfName> hcclSocketIfName{"HCCL_SOCKET_IFNAME", SocketIfName({}, false, false), CastSocketIfName};
    CfgField<bool> whitelistDisable{"HCCL_WHITELIST_DISABLE", true, CastBin2Bool};
    CfgField<std::string> hcclWhiteListFile{"HCCL_WHITELIST_FILE", "", Str2T<std::string>, CheckFilePath, SetRealPath};
    CfgField<std::vector<SocketPortRange>> hcclHostSocketPortRange{
        "HCCL_HOST_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_HOST_SOCKET_PORT_RANGE");
        }};
    CfgField<std::vector<SocketPortRange>> hcclDeviceSocketPortRange{
        "HCCL_NPU_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
        }};
};

// Socket公共配置
class EnvSocketConfig : public BaseConfig {
public:
    void Parse() override;
    s32 GetSocketFamily() const;
    s32 GetLinkTimeOut() const;

private:
    static constexpr s32 HCCL_LINK_TIME_OUT_S = 120;            // HCCL 默认的建链超时时间设置为120s
    static constexpr s32 HCCL_MIN_LINK_TIME_OUT_S = 120;        // HCCL 建链最小超时时间设置为120s
    static constexpr s32 HCCL_MAX_LINK_TIME_OUT_S = (120 * 60); // HCCL 最大建链超时时间设置为120*60s

    CfgField<s32> hcclSocketFamily{"HCCL_SOCKET_FAMILY", -1, CastSocketFamily};
    CfgField<s32> linkTimeOut{
        "HCCL_CONNECT_TIMEOUT", s32(HCCL_LINK_TIME_OUT_S), Str2T<s32>,
        CHK_RANGE_CLOSED<s32>(HCCL_MIN_LINK_TIME_OUT_S, HCCL_MAX_LINK_TIME_OUT_S)};
};

// RTS配置
class EnvRtsConfig : public BaseConfig {
public:
    void Parse() override;
    u32 GetExecTimeOut() const;
    double GetAivExecTimeOut() const;

private:
    static constexpr s32 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68; // notifywait默认1836等待时长
    static constexpr s32 AIV_TIMEOUT_DEFAULT = 1091;

    CfgField<u32> execTimeOut{
        "HCCL_EXEC_TIMEOUT", static_cast<u32>(NOTIFY_DEFAULT_WAIT_TIME),
        [](const std::string& s) -> u32 {
            static std::regex validFormat(R"(^\d+(\.\d{1,2})?$)");
            if (!std::regex_match(s, validFormat)) {
                THROW<InvalidParamsException>(
                    StringFormat("Invalid config value, execTimeOutStr[%s], up to two decimal places", s.c_str()));
            }
            return String2T<u32>(s);
        },
        CheckExecTimeOut, ProcExecTimeOut};

    CfgField<double> aivExecTimeOut{
        "HCCL_EXEC_TIMEOUT", double(AIV_TIMEOUT_DEFAULT),
        [](const std::string& s) -> double {
            static std::regex validFormat(R"(^\d+(\.\d{1,2})?$)");
            if (!std::regex_match(s, validFormat)) {
                THROW<InvalidParamsException>(
                    StringFormat("Invalid config value, execTimeOutStr[%s], up to two decimal places", s.c_str()));
            }
            return String2T<double>(s);
        },
        nullptr, nullptr};
};

// RDMA配置
class EnvRdmaConfig : public BaseConfig {
public:
    void Parse() override;
    u32 GetRdmaTrafficClass() const;
    u32 GetRdmaServerLevel() const;
    u32 GetRdmaTimeOut() const;
    u32 GetRdmaRetryCnt() const;
    u32 GetRdmaQueueNum() const;
    u32 GetRdmaMultiQpThreshold() const;
    const HostRdmaUdpPortsList& GetHostRdmaUdpPortsList() const;
    const MultiQpSrcPortConfig& GetMultiQpSrcPortConfig() const;

    static constexpr u32 HCCL_RDMA_TC_DEFAULT = 132;      // 默认的traffic class为132(33*4)
    static constexpr u32 HCCL_RDMA_SL_DEFAULT = 4;        // 默认的server level为4
    static constexpr u32 HCCL_RDMA_TIMEOUT_DEFAULT = 20;  // 默认的TIMEOUT配置为20(对应时间4.096*2^20us)
    static constexpr u32 HCCL_RDMA_RETRY_CNT_DEFAULT = 7; // 默认的Retry Cnt为7
    static constexpr u32 HCCL_RDMA_TC_MIN = 0;            // rdma traffic class最小值为0
    static constexpr u32 HCCL_RDMA_TC_MAX = 255;          // rdma traffic class最大值为255
    static constexpr u32 HCCL_RDMA_SL_MIN = 0;            // rdma server level最小值为0
    static constexpr u32 HCCL_RDMA_SL_MAX = 7;            // rdma server level最大值为7
    static constexpr u32 HCCL_RDMA_TIMEOUT_MIN = 0;       // rdma timeout最小值为0
    static constexpr u32 HCCL_RDMA_TIMEOUT_MAX = 31;      // rdma timeout最大值为31
    static constexpr u32 HCCL_RDMA_RETRY_CNT_MIN = 1;     // rdma Retry Cnt最小值为1
    static constexpr u32 HCCL_RDMA_RETRY_CNT_MAX = 7;     // rdma Retry Cnt最大值为7

private:
    CfgField<u32> rdmaTrafficClass{
        "HCCL_RDMA_TC", u32(HCCL_RDMA_TC_DEFAULT), Str2T<u32>,
        CHK_RANGE_CLOSED<u32>(HCCL_RDMA_TC_MIN, HCCL_RDMA_TC_MAX), CheckRDMATrafficClass};
    CfgField<u32> rdmaServerLevel{
        "HCCL_RDMA_SL", u32(HCCL_RDMA_SL_DEFAULT), Str2T<u32>,
        CHK_RANGE_CLOSED<u32>(HCCL_RDMA_SL_MIN, HCCL_RDMA_SL_MAX)};
    CfgField<u32> rdmaTimeOut{
        "HCCL_RDMA_TIMEOUT", u32(HCCL_RDMA_TIMEOUT_DEFAULT), Str2T<u32>, CheckRdmaTimeout, ProcRdmaTimeout};
    CfgField<u32> rdmaRetryCnt{
        "HCCL_RDMA_RETRY_CNT", u32(HCCL_RDMA_RETRY_CNT_DEFAULT), Str2T<u32>,
        CHK_RANGE_CLOSED<u32>(HCCL_RDMA_RETRY_CNT_MIN, HCCL_RDMA_RETRY_CNT_MAX)};
    CfgField<u32> queueNum{"HCCL_RDMA_QPS_PER_CONNECTION", u32(1), Str2T<u32>, CHK_RANGE_CLOSED<u32>(1, 32)};
    CfgField<u32> multiQpThreshold{
        "HCCL_MULTI_QP_THRESHOLD", u32(512 * 1024), Str2T<u32>, CHK_RANGE_CLOSED<u32>(1, 8192), ConvertUnitQpThreshold};
    void ParseMultiQpSrcPortConfig();
    HcclResult OpenMultiQpConfigFile(std::ifstream& inFile);
    HcclResult ParseConfigContent(std::ifstream& inFile, MultiQpSrcPortConfig& config);
    HcclResult ParseLineToIpPairAndPortPart(
        const std::string& lineInfo, u32 lineCnt, const std::string& lineAvator, std::string& ipPairKey,
        std::string& portPart);
    HcclResult ParseSrcPortsFromPortPart(
        const std::string& portPart, u32 lineCnt, const std::string& lineAvator, std::vector<std::uint16_t>& ports);
    void LogMultiQpSrcPortConfig() const;
    CfgField<std::string> qpPortConfigPath{
        "HCCL_RDMA_QP_PORT_CONFIG_PATH", "", Str2T<std::string>, CheckFilePath, SetRealPath};
    MultiQpSrcPortConfig multiQpSrcPortConfig_;
    CfgField<HostRdmaUdpPortsList> hostRdmaUdpPortsList{"HCCL_HOST_RDMA_UDP_PORTS_LIST", {}, CastHostRdmaUdpPortsList};
};

// 算法配置
class EnvAlgoConfig : public BaseConfig {
public:
    void Parse() override;
    const std::string& GetPrimQueueGenName() const;
    const std::map<OpType, std::vector<HcclAlgoType>> GetAlgoConfig() const;
    u64 GetBuffSize() const;
    HcclAccelerator GetHcclAccelerator() const;
    bool GetDeterministic() const;

private:
    static constexpr u32 HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE = 200;
    static constexpr u32 HCCL_CCL_COMM_BUFFER_MIN = 1;
    static constexpr u64 HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE = (1 * 1024 * 1024);

    CfgField<std::string> primQueueGenName{"PRIM_QUEUE_GEN_NAME", "", Str2T<std::string>};

    CfgField<std::map<OpType, std::vector<HcclAlgoType>>> hcclAlgoConfig{
        "HCCL_ALGO", std::map<OpType, std::vector<HcclAlgoType>>(), SetHcclAlgoConfig};

    CfgField<u64> bufferSize{
        "HCCL_BUFFSIZE", HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE* HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE, Str2T<u64>,
        CHK_RANGE_CLOSED<u64>(HCCL_CCL_COMM_BUFFER_MIN, ULLONG_MAX), [](u64& i) {
            i *= HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE;
        }};
    CfgField<HcclAccelerator> hcclAccelerator_{
        "HCCL_OP_EXPANSION_MODE", HcclAccelerator::AICPU_TS, CastHcclAccelerator};
};

// 日志/DFX配置
class EnvLogConfig : public BaseConfig {
public:
    void Parse() override;
    bool GetEntryLogEnable() const;
    const std::string& GetCannVersion() const;
    const DfsConfig& GetDfsConfig() const;

private:
    CfgField<bool> entryLogEnable{"HCCL_ENTRY_LOG_ENABLE", false, CastBin2Bool};
    CfgField<std::string> cannVersion{"LD_LIBRARY_PATH", "", CastCannVersion};
    CfgField<DfsConfig> dfsConfig{"HCCL_DFS_CONFIG", DfsConfig(true, true, 0), CastDfsConfig};
};

// 绕路使能环境变量
class EnvDetourConfig : public BaseConfig {
public:
    void Parse() override;
    virtual HcclDetourType GetDetourType() const;

private:
    CfgField<HcclDetourType> detourType{"HCCL_DETOUR", HcclDetourType::HCCL_DETOUR_DISABLE, CastDetourType};
};

} // namespace Hccl

#endif // HCCLV2_BASE_CONFIG_LEGACY_H
