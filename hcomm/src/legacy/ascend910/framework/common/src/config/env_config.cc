/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_config.h"
#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include "adapter_error_manager_pub.h"
#include "log.h"
#include "sal_pub.h"
#include "mmpa_api.h"
#include "config_log.h"
#include "adapter_rts_common.h"

using namespace hccl;

static std::mutex g_envConfigMutex;

constexpr char ENV_EMPTY_STRING[] = "EmptyString";

constexpr char HCCL_AUTO_PORT_CONFIG[] = "auto"; // 端口范围配置为auto时，由OS分配浮动监听端口
constexpr u32 MAX_PORT_NUMBER = 65535;           // 合法端口号的上限
constexpr u32 HCCL_SOCKET_PORT_RANGE_AUTO = 0;   // 需要保留的
const std::string CLUSTER_HEART_CONFIG = "cluster_heartbeat:";
const std::string STUCK_DETECTION_CONFIG = "stuck_detection:";
const std::string INCONSISTENT_CHECK_CONFIG = "inconsistent_check:";
const std::string CONNECTION_FAULT_DETECTION_TIME = "connection_fault_detection_time:";
const std::string TASK_MONITOR_INTERVAL = "task_monitor_interval:";
constexpr static const s32 HCCL_MAX_LINK_TIME_OUT_S = (120 * 60); // HCCL 最大探测超时时间设置为120*60s
HcclResult InitEnvConfig()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    if (g_envConfig.initialized) {
        return HCCL_SUCCESS;
    }
    // 初始化环境变量
    CHK_RET(InitEnvParam());

    g_envConfig.initialized = true;

    return HCCL_SUCCESS;
}

bool GetExternalInputHostPortSwitch()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.hostSocketPortSwitch;
}

bool GetExternalInputNpuPortSwitch()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.npuSocketPortSwitch;
}

const u32& EnvConfig::GetExternalInputRdmaTrafficClass() { return g_envConfig.rdmaTrafficClass; }

const u32& EnvConfig::GetExternalInputRdmaServerLevel() { return g_envConfig.rdmaServerLevel; }

const u32& EnvConfig::GetExternalInputRdmaTimeOut() { return g_envConfig.rdmaTimeOut; }

const u32& EnvConfig::GetExternalInputRdmaRetryCnt() { return g_envConfig.rdmaRetryCnt; }

const std::vector<HcclSocketPortRange>& GetExternalInputHostSocketPortRange()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.hostSocketPortRange;
}

const std::vector<HcclSocketPortRange>& GetExternalInputNpuSocketPortRange()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.npuSocketPortRange;
}

s32& GetExternalInputDfsConnectionFaultDetectionTime() { return g_envConfig.dfsConnectionFaultDetectionTime; }

u32& GetExternalInputDfsTaskMonitorInterval() { return g_envConfig.dfsTaskMonitorInterval; }

HcclResult ResetEnvConfigInitState()
{
    g_envConfig.SetDefaultParams();
    return HCCL_SUCCESS;
}

HcclResult InitEnvParam()
{
    HcclResult ret = ParseHostSocketPortRange();
    char* mmSysGetHostEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_HOST_SOCKET_PORT_RANGE, mmSysGetHostEnvValue);
    std::string hostSocketPortRangeEnv = (mmSysGetHostEnvValue != nullptr) ? mmSysGetHostEnvValue : "EmptyString";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {hostSocketPortRangeEnv, "HCCL_HOST_SOCKET_PORT_RANGE",
             "a valid port range in the format of port1-port2, where port1 and port2 are valid port numbers "
             "(0-65535)"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init environment param, parse "
            "HCCL_HOST_SOCKET_PORT_RANGE failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = ParseNpuSocketPortRange();
    char* mmSysGetNpuEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_NPU_SOCKET_PORT_RANGE, mmSysGetNpuEnvValue);
    std::string npuSocketPortRangeEnv = (mmSysGetNpuEnvValue != nullptr) ? mmSysGetNpuEnvValue : "EmptyString";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {npuSocketPortRangeEnv, "HCCL_NPU_SOCKET_PORT_RANGE",
             "a valid port range in the format of port1-port2, where port1 and port2 are valid port numbers "
             "(0-65535)"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init environment param, parse "
            "HCCL_NPU_SOCKET_PORT_RANGE failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = ParseDFSConfig();
    char* dfsConfigValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_DFS_CONFIG, dfsConfigValue);
    std::string dfsConfigEnv = (dfsConfigValue != nullptr) ? dfsConfigValue : "EmptyString";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {dfsConfigEnv, "HCCL_DFS_CONFIG",
             "comma-separated key-value pairs (e.g., cluter_heartbeat=on,stuck_detection=off)"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init environment param, parse "
            "HCCL_DFS_CONFIG failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = g_envConfig.ParseRDMATrafficClass();
    char* mmSysGetTrafficClassEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_TC, mmSysGetTrafficClassEnvValue);
    std::string mmSysGetTrafficClassEnv
        = (mmSysGetTrafficClassEnvValue != nullptr) ? mmSysGetTrafficClassEnvValue : "EmptyString";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({mmSysGetTrafficClassEnv, "HCCL_RDMA_TC", "range[0, 255], Must be a multiple of 4"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init environment param, parse "
            "HCCL_RDMA_TC failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = g_envConfig.ParseRDMAServerLevel();
    char* mmSysGetServerLevelEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_SL, mmSysGetServerLevelEnvValue);
    std::string mmSysGetServerLevel
        = (mmSysGetServerLevelEnvValue != nullptr) ? mmSysGetServerLevelEnvValue : "EmptyString";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({mmSysGetServerLevel, "HCCL_RDMA_SL", "Value range[0, 7]"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init environment param, parse "
            "HCCL_RDMA_SL failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析RDMATimeOut
    std::pair<u32, u32> rdmaTimeOutRange;
    ret = g_envConfig.ParseRDMATimeOut(rdmaTimeOutRange);
    char* mmSysGetTimeOutEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_TIMEOUT, mmSysGetTimeOutEnvValue);
    std::string timeOutEnv = (mmSysGetTimeOutEnvValue != nullptr) ? mmSysGetTimeOutEnvValue : "EmptyString";
    std::string vaildRange
        = "range[" + std::to_string(rdmaTimeOutRange.first) + " ," + std::to_string(rdmaTimeOutRange.second) + "]";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({timeOutEnv, "HCCL_RDMA_TIMEOUT", vaildRange}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_RDMA_TIMEOUT failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析RDMARetryCnt
    ret = g_envConfig.ParseRDMARetryCnt();
    char* mmSysGetRetryCntEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_RETRY_CNT, mmSysGetRetryCntEnvValue);
    std::string retryCntEnv = (mmSysGetRetryCntEnvValue != nullptr) ? mmSysGetRetryCntEnvValue : "EmptyString";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({retryCntEnv, "HCCL_RDMA_RETRY_CNT", "range[1, 7]"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse HCCL_RDMA_RETRY_CNT failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    ret = InitDebugConfigByEnv();
    char* env = nullptr; // 环境变量值
    MM_SYS_GET_ENV(MM_ENV_HCCL_DEBUG_CONFIG, env);
    std::string envValue = env ? std::string(env) : "null";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>({envValue, "HCCL_DEBUG_CONFIG", "ALG,TASK,RESOURCE(optionally prefixed with'^'"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init environment param, parse "
            "HCCL_DEBUG_CONFIG failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);

    // 解析算法配置
    ret = ParseHcclAlgo();
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_ALGO, mmSysGetEnvValue);
    std::string hcclAlgo = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    RPT_ENV_ERR(
        ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {hcclAlgo, "HCCL_ALGO",
             "level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] In init env variable param, parse "
            "hccl algorithm config failed. errorno[%d]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), HCCL_ERROR_CODE(ret), ret),
        ret);
    return HCCL_SUCCESS;
}

bool EnvConfig::CheckEnvLen(const char* envStr, u32 envMaxLen)
{
    // 校验环境变量长度
    u32 envLen = strnlen(envStr, envMaxLen + 1);
    if (envLen == (envMaxLen + 1)) {
        HCCL_ERROR(
            "[CheckEnvLen] errNo[0x%016llx] env len is invalid, len is %u", HCCL_ERROR_CODE(HCCL_E_PARA), envLen);
        return false;
    }
    return true;
}

HcclResult SetDefaultSocketPortRange(const SocketLocation& socketLoc, std::vector<HcclSocketPortRange>& portRangeVec)
{
    if (socketLoc == SOCKET_HOST) {
        g_envConfig.hostSocketPortSwitch = false;
        portRangeVec.clear();
        HCCL_RUN_WARNING("[HCCL_ENV] HCCL_HOST_SOCKET_PORT_RANGE is not set. Multi-process will not be supported!");
    } else if (socketLoc == SOCKET_NPU) {
        g_envConfig.npuSocketPortSwitch = false;
        portRangeVec.clear();
        HCCL_RUN_WARNING("[HCCL_ENV] HCCL_NPU_SOCKET_PORT_RANGE is not set. Multi-process will not be supported!");
    } else {
        HCCL_ERROR("[SetDefaultSocketPortRange] undefined socket location, fail to init socket port range by default.");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckSocketPortRangeValid(const std::string& envName, const std::vector<HcclSocketPortRange>& portRanges)
{
    std::vector<HcclSocketPortRange> rangeVec(portRanges.begin(), portRanges.end());
    std::sort(rangeVec.begin(), rangeVec.end(), [](auto& a, auto& b) {
        return a.min == b.min ? a.max < b.max : a.min < b.min;
    });
    for (size_t i = 0; i < rangeVec.size(); ++i) {
        // the socket range should not be inverted
        CHK_PRT_RET(
            rangeVec[i].min > rangeVec[i].max,
            HCCL_ERROR(
                "[Check][PortRangeValid] In %s, in socket port range [%u, %u], "
                "the lower bound is greater than the upper bound.",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max),
            HCCL_E_PARA);

        // the socket range should not include the reserved port for auto listening.
        CHK_PRT_RET(
            rangeVec[i].min <= HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_ERROR(
                "[Check][PortRangeValid] In %s, socket port range [%u, %u] includes "
                "the reserved port number [%u]. please do not use port [%u] in socket port range.",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max, HCCL_SOCKET_PORT_RANGE_AUTO,
                HCCL_SOCKET_PORT_RANGE_AUTO),
            HCCL_E_PARA);

        // the socket range should not exceed the maximum port number
        CHK_PRT_RET(
            rangeVec[i].max > MAX_PORT_NUMBER,
            HCCL_ERROR(
                "[Check][PortRangeValid] In %s, in socket port range [%u, %u], "
                "the upper bound exceed max port number[%u].",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max, MAX_PORT_NUMBER),
            HCCL_E_PARA);

        // the socket range should not be overlapped
        CHK_PRT_RET(
            i != 0 && rangeVec[i - 1].max >= rangeVec[i].min,
            HCCL_ERROR(
                "[Check][PortRangeValid] In %s, "
                "socket port range [%u, %u] is conflict with socket port range [%u, %u].",
                envName.c_str(), rangeVec[i - 1].min, rangeVec[i - 1].max, rangeVec[i].min, rangeVec[i].max),
            HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}

HcclResult GetUIntFromStr(const std::string& digitStr, u32& val)
{
    HcclResult ret = IsAllDigit(digitStr.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetUIntFromStr] str[%s] is not all digit.", digitStr.c_str()), ret);
    ret = SalStrToULong(digitStr.c_str(), HCCL_BASE_DECIMAL, val);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[GetUIntFromStr] str[%s] is a invalid number.", digitStr.c_str()), ret);
    return HCCL_SUCCESS;
}

bool SplitString(std::string& totalStr, std::string& prefixStr, const std::string& delim)
{
    std::size_t found = totalStr.find(delim);
    if (found == std::string::npos) {
        return false;
    }
    prefixStr = totalStr.substr(0, found);
    totalStr = totalStr.substr(found + 1);
    return true;
}

HcclResult SplitSinglePortRange(const std::string& envName, std::string& rangeStr, HcclSocketPortRange& portRange)
{
    std::string rangeMin{};
    const std::string delim = "-";
    if (SplitString(rangeStr, rangeMin, delim)) {
        CHK_RET(GetUIntFromStr(rangeMin, portRange.min));
        CHK_RET(GetUIntFromStr(rangeStr, portRange.max));
    } else {
        CHK_RET(GetUIntFromStr(rangeStr, portRange.min));
        portRange.max = portRange.min;
    }
    HCCL_INFO(
        "[Split][SinglePortRange] Load hccl socket port range [%u, %u] from %s", portRange.min, portRange.max,
        envName.c_str());
    return HCCL_SUCCESS;
}

HcclResult SplitHcclSocketPortRange(
    const std::string& envName, std::string& portRangeConfig, std::vector<HcclSocketPortRange>& portRangeVec)
{
    std::string rangeStr{};
    const std::string delim = ",";
    while (SplitString(portRangeConfig, rangeStr, delim)) {
        HcclSocketPortRange portRange = {};
        CHK_RET(SplitSinglePortRange(envName, rangeStr, portRange));
        portRangeVec.emplace_back(portRange);
    }
    HcclSocketPortRange portRange = {};
    CHK_RET(SplitSinglePortRange(envName, portRangeConfig, portRange));
    portRangeVec.emplace_back(portRange);

    CHK_RET(CheckSocketPortRangeValid(envName, portRangeVec));
    return HCCL_SUCCESS;
}

HcclResult PortRangeSwitchOn(const SocketLocation& socketLoc)
{
    if (socketLoc == SOCKET_HOST) {
        g_envConfig.hostSocketPortSwitch = true;
        HCCL_INFO("HCCL_HOST_SOCKET_PORT_RANGE is set, switch on.");
    } else if (socketLoc == SOCKET_NPU) {
        g_envConfig.npuSocketPortSwitch = true;
        HCCL_INFO("HCCL_NPU_SOCKET_PORT_RANGE is set, switch on.");
    } else {
        HCCL_ERROR("[PortRangeSwitchOn] undefined socket location, fail to init socket port range by default.");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

void PrintSocketPortRange(const std::string& envName, const std::vector<HcclSocketPortRange>& portRangeVec)
{
    // assemble port ranges into a string to print the result range
    std::ostringstream portRangeOss;
    for (auto range : portRangeVec) {
        portRangeOss << " [" << std::to_string(range.min) << ", " << std::to_string(range.max) << "]";
    }
    HCCL_RUN_INFO("%s is set to%s.", envName.c_str(), portRangeOss.str().c_str());
}

HcclResult SetSocketPortRange(
    const std::string& envName, const std::string& socketPortRange, const SocketLocation& socketLoc,
    std::vector<HcclSocketPortRange>& portRangeVec)
{
    portRangeVec.clear();

    // the environment variable is not set
    if (socketPortRange.compare(ENV_EMPTY_STRING) == 0) {
        CHK_RET(SetDefaultSocketPortRange(socketLoc, portRangeVec));
        return HCCL_SUCCESS;
    }

    // the socket port range is set to auto, then the os will listen on the ports dynamically and automatically.
    if (socketPortRange.compare(HCCL_AUTO_PORT_CONFIG) == 0) {
        HcclSocketPortRange autoSocketPortRange = {HCCL_SOCKET_PORT_RANGE_AUTO, HCCL_SOCKET_PORT_RANGE_AUTO};
        portRangeVec.emplace_back(autoSocketPortRange);
        CHK_RET(PortRangeSwitchOn(socketLoc));
        HCCL_RUN_INFO(
            "%s is set to %s as [%u, %u].", envName.c_str(), HCCL_AUTO_PORT_CONFIG, autoSocketPortRange.min,
            autoSocketPortRange.max);
        return HCCL_SUCCESS;
    }

    std::string portRangeConfig = socketPortRange;
    // the environment variable is set to an empty string
    portRangeConfig.erase(std::remove(portRangeConfig.begin(), portRangeConfig.end(), ' '), portRangeConfig.end());
    if (portRangeConfig.empty()) {
        CHK_RET(SetDefaultSocketPortRange(socketLoc, portRangeVec));
        return HCCL_SUCCESS;
    }
    // load ranges from string
    CHK_RET(SplitHcclSocketPortRange(envName, portRangeConfig, portRangeVec));
    if (portRangeVec.size() == 0) {
        HCCL_ERROR("Load empty port range from %s, please check.", envName.c_str());
        return HCCL_E_PARA;
    }
    CHK_RET(PortRangeSwitchOn(socketLoc));
    (void)PrintSocketPortRange(envName, portRangeVec);
    return HCCL_SUCCESS;
}

HcclResult ParseHostSocketPortRange()
{
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910 && deviceType != DevType::DEV_TYPE_910B
        && deviceType != DevType::DEV_TYPE_910_93) {
        g_envConfig.hostSocketPortSwitch = false;
        g_envConfig.hostSocketPortRange.clear();
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_HOST_SOCKET_PORT_RANGE is not supported on devType[%u], nothing is loaded.", deviceType);
        return HCCL_SUCCESS;
    }

    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_HOST_SOCKET_PORT_RANGE, mmSysGetEnvValue);
    std::string hostSocketPortRangeEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    CHK_RET(SetSocketPortRange(
        "HCCL_HOST_SOCKET_PORT_RANGE", hostSocketPortRangeEnv, SOCKET_HOST, g_envConfig.hostSocketPortRange));
    return HCCL_SUCCESS;
}

HcclResult ParseNpuSocketPortRange()
{
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910 && deviceType != DevType::DEV_TYPE_910B
        && deviceType != DevType::DEV_TYPE_910_93) {
        g_envConfig.npuSocketPortSwitch = false;
        g_envConfig.npuSocketPortRange.clear();
        HCCL_RUN_INFO(
            "[HCCL_ENV] HCCL_NPU_SOCKET_PORT_RANGE is not supported on devType[%u], nothing is loaded.", deviceType);
        return HCCL_SUCCESS;
    }
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_NPU_SOCKET_PORT_RANGE, mmSysGetEnvValue);
    std::string npuSocketPortRangeEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    CHK_RET(SetSocketPortRange(
        "HCCL_NPU_SOCKET_PORT_RANGE", npuSocketPortRangeEnv, SOCKET_NPU, g_envConfig.npuSocketPortRange));
    return HCCL_SUCCESS;
}

// 通用的环境变量解析函数
HcclResult ParseEnvConfig(const EnvConfigParam& param, std::string& envValue, u32& resultValue)
{
    if (envValue.compare(ENV_EMPTY_STRING) == 0) {
        HCCL_RUN_INFO("%s set by default to [%u]", param.envName.c_str(), param.defaultValue);
        resultValue = param.defaultValue;
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = g_envConfig.CheckEnvLen(envValue.c_str(), g_envConfig.MAX_LEN_OF_DIGIT_ENV);
    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][%s] errNo[0x%016llx] Invalid %s env len, len is bigger than [%u], errorno[%d]",
            param.envName.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA), param.envName.c_str(),
            g_envConfig.MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    CHK_RET(IsAllDigit(envValue.c_str()));

    HcclResult ret = SalStrToULong(envValue.c_str(), HCCL_BASE_DECIMAL, resultValue);
    // 若转换出错或者设置的值不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || resultValue < param.minValue || resultValue > param.maxValue),
        HCCL_ERROR(
            "[Parse][%s] is invalid. expect: [%u, %u], actual: [%u]", param.envName.c_str(), param.minValue,
            param.maxValue, resultValue),
        HCCL_E_PARA);

    // 如果提供了baseValue，检查是否是baseValue的整数倍
    if (param.baseValue != 0 && resultValue % param.baseValue != 0) {
        HCCL_ERROR("[Parse] %s[%u] is not a multiple of [%u]", param.envName.c_str(), resultValue, param.baseValue);
        return HCCL_E_PARA;
    }

    HCCL_RUN_INFO("%s set by environment to [%u]", param.envName.c_str(), resultValue);
    return HCCL_SUCCESS;
}

HcclResult EnvConfig::ParseRDMATrafficClass()
{
    EnvConfigParam param
        = {"HCCL_RDMA_TC", HCCL_RDMA_TC_DEFAULT, HCCL_RDMA_TC_MIN, HCCL_RDMA_TC_MAX, HCCL_RDMA_TC_BASE};
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_TC, mmSysGetEnvValue);
    std::string envValue = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    return ParseEnvConfig(param, envValue, g_envConfig.rdmaTrafficClass);
}

HcclResult EnvConfig::ParseRDMAServerLevel()
{
    EnvConfigParam param = {"HCCL_RDMA_SL", HCCL_RDMA_SL_DEFAULT, HCCL_RDMA_SL_MIN, HCCL_RDMA_SL_MAX, 0};
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_SL, mmSysGetEnvValue);
    std::string envValue = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    return ParseEnvConfig(param, envValue, g_envConfig.rdmaServerLevel);
}

HcclResult EnvConfig::ParseRDMATimeOut(std::pair<u32, u32>& rdmaTimeOutRange)
{
    u32 rdmaTimeOutMax;
#ifndef HCCD
    if (!IsGeneralServer()) {
        DevType deviceType;
        CHK_RET(hrtGetDeviceType(deviceType));
        rdmaTimeOutMax = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) ?
                             HCCL_RDMA_TIMEOUT_MAX_910_93 :
                             HCCL_RDMA_TIMEOUT_MAX;
    } else {
        rdmaTimeOutMax = HCCL_RDMA_TIMEOUT_MAX;
    }
#else
    rdmaTimeOutMax = HCCL_RDMA_TIMEOUT_MAX;
#endif
    rdmaTimeOutRange.first = HCCL_RDMA_TIMEOUT_MIN;
    rdmaTimeOutRange.second = rdmaTimeOutMax;
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_TIMEOUT, mmSysGetEnvValue);
    std::string timeOutEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    u32 rdmaTimeOut = HCCL_RDMA_TIMEOUT_DEFAULT;
    if (timeOutEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TIMEOUT set by default to [%u]", rdmaTimeOut);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(timeOutEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);

    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][RDMATimeOut]errNo[0x%016llx] Invalid HCCL_RDMA_TIMEOUT env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    g_envConfig.rdmaTimeOut = HCCL_RDMA_TIMEOUT_DEFAULT;
    CHK_RET(IsAllDigit(timeOutEnv.c_str()));

    HcclResult ret = SalStrToULong(timeOutEnv.c_str(), HCCL_BASE_DECIMAL, rdmaTimeOut);
    // 若转换出错或者设置的RDMATimeOut不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || rdmaTimeOut < HCCL_RDMA_TIMEOUT_MIN || rdmaTimeOut > rdmaTimeOutMax),
        HCCL_ERROR(
            "[Parse][RDMATimeOut]HCCL_RDMA_TIMEOUT[%s] is invalid. expect: [%u, %u]", timeOutEnv.c_str(),
            HCCL_RDMA_TIMEOUT_MIN, rdmaTimeOutMax),
        HCCL_E_PARA);

    g_envConfig.rdmaTimeOut = rdmaTimeOut;
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_TIMEOUT set by environment to [%u]", rdmaTimeOut);
    return HCCL_SUCCESS;
}

HcclResult EnvConfig::ParseRDMARetryCnt()
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_RETRY_CNT, mmSysGetEnvValue);
    std::string retryCntEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    u32 rdmaRetryCnt = HCCL_RDMA_RETRY_CNT_DEFAULT;
    if (retryCntEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_RETRY_CNT set by default to [%u]", rdmaRetryCnt);
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = CheckEnvLen(retryCntEnv.c_str(), MAX_LEN_OF_DIGIT_ENV);

    CHK_PRT_RET(
        !isEnvLenValid,
        HCCL_ERROR(
            "[Parse][rdmaRetryCnt]errNo[0x%016llx] Invalid HCCL_RDMA_RETRY_CNT env len, len is bigger than "
            "[%u]. errorno[%d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);

    g_envConfig.rdmaRetryCnt = HCCL_RDMA_RETRY_CNT_DEFAULT;
    CHK_RET(IsAllDigit(retryCntEnv.c_str()));

    HcclResult ret = SalStrToULong(retryCntEnv.c_str(), HCCL_BASE_DECIMAL, rdmaRetryCnt);
    // 若转换出错或者设置的RDMARetryCnt不在有效范围内，报错
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS || rdmaRetryCnt < HCCL_RDMA_RETRY_CNT_MIN || rdmaRetryCnt > HCCL_RDMA_RETRY_CNT_MAX),
        HCCL_ERROR(
            "[Parse][rdmaRetryCnt]HCCL_RDMA_RETRY_CNT[%s] is invalid. expect: [%u, %u]", retryCntEnv.c_str(),
            HCCL_RDMA_RETRY_CNT_MIN, HCCL_RDMA_RETRY_CNT_MAX),
        HCCL_E_PARA);
    g_envConfig.rdmaRetryCnt = rdmaRetryCnt;
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_RDMA_RETRY_CNT set by environment to [%u]", rdmaRetryCnt);
    return HCCL_SUCCESS;
}

HcclResult
ParseSingleDFSConfigItem(const std::string& dfsConfigEnv, const std::string& configName, std::string& configResult)
{
    size_t start = dfsConfigEnv.find(configName);
    if (start == std::string::npos) {
        HCCL_INFO("[Parse] DFS config item [%s] is not found.", configName.c_str());
        return HCCL_SUCCESS;
    }
    size_t end = dfsConfigEnv.find(",", start);
    if (end == std::string::npos) {
        configResult = dfsConfigEnv.substr(start + configName.size());
    } else {
        configResult = dfsConfigEnv.substr(start + configName.size(), end - start - configName.size());
    }
    HCCL_INFO("[Parse] DFS config item %s [%s]", configName.c_str(), configResult.c_str());
    return HCCL_SUCCESS;
}

HcclResult ParseMonitor(std::string& taskMonitorInterval, s32& monitorTime)
{
    if (taskMonitorInterval.empty()) {
        monitorTime = 0;
    } else {
        HcclResult ret = SalStrToInt(taskMonitorInterval, HCCL_BASE_DECIMAL, monitorTime);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ParseDFSConfig] HCCL_DFS_CONFIG-task_monitor_interval[%s]"
                "is invalid, errorno[%d]",
                taskMonitorInterval.c_str(), ret),
            ret);
    }

    s32 maxTimeInMs = HCCL_MAX_LINK_TIME_OUT_S * 1000;
    if (monitorTime == 0) {
        g_envConfig.dfsTaskMonitorInterval = 0;
    } else if (monitorTime >= 0 && monitorTime <= maxTimeInMs) {
        g_envConfig.dfsTaskMonitorInterval = monitorTime;
    } else { // 不在允许范围内报错
        HCCL_ERROR(
            "[ParseDFSConfig] HCCL_DFS_CONFIG-task_monitor_interval[%d ms] is invalid, expect: [0, %d ms]", monitorTime,
            maxTimeInMs);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ParseDFSConfig()
{
    char* dfsConfigValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_DFS_CONFIG, dfsConfigValue);
    std::string dfsConfigEnv = (dfsConfigValue != nullptr) ? dfsConfigValue : "EmptyString";
    if (dfsConfigEnv == "EmptyString") {
        HCCL_RUN_INFO("[HCCL_ENV][Parse][HCCL_DFS_CONFIG] Parse environmental variable HCCL_DFS_CONFIG is not set.");
        return HCCL_SUCCESS;
    }

    // 去除空格
    dfsConfigEnv.erase(std::remove(dfsConfigEnv.begin(), dfsConfigEnv.end(), ' '), dfsConfigEnv.end());

    std::transform(dfsConfigEnv.begin(), dfsConfigEnv.end(), dfsConfigEnv.begin(), ::tolower);

    std::string heartbeatSwitch;
    CHK_RET(ParseSingleDFSConfigItem(dfsConfigEnv, CLUSTER_HEART_CONFIG, heartbeatSwitch));
    if (heartbeatSwitch == "off") {
        g_envConfig.enableClusterHeartBeat = false;
    } else if (heartbeatSwitch == "on") {
        g_envConfig.enableClusterHeartBeat = true;
    } else {
        HCCL_RUN_WARNING(
            "[HCCL_ENV][ParseDFSConfig] HCCL_DFS_CONFIG-cluster_heartbeat was configured to [%s], please configure to "
            "'on' or 'off'",
            heartbeatSwitch.c_str());
    }

    std::string stuckDetectSwitch;
    CHK_RET(ParseSingleDFSConfigItem(dfsConfigEnv, STUCK_DETECTION_CONFIG, stuckDetectSwitch));
    if (stuckDetectSwitch == "off") {
        g_envConfig.opCounterEnable = false;
    } else if (stuckDetectSwitch == "on") {
        g_envConfig.opCounterEnable = true;
    } else {
        HCCL_RUN_WARNING(
            "[HCCL_ENV][ParseDFSConfig] HCCL_DFS_CONFIG-stuck_detection was configured to [%s], please configure to "
            "'on' or 'off'",
            stuckDetectSwitch.c_str());
    }

    // 解析算子不一致故障检测能力开关
    std::string inconsistentCheckSwitch;
    CHK_RET(ParseSingleDFSConfigItem(dfsConfigEnv, INCONSISTENT_CHECK_CONFIG, inconsistentCheckSwitch));
    if (inconsistentCheckSwitch == "off") {
        g_envConfig.inconsistentCheckSwitch = InconsistentCheckMode::OFF;
    } else if (inconsistentCheckSwitch == "on") {
        g_envConfig.inconsistentCheckSwitch = InconsistentCheckMode::ON;
    } else if (inconsistentCheckSwitch == "first") {
        g_envConfig.inconsistentCheckSwitch = InconsistentCheckMode::FIRST;
    } else {
        HCCL_RUN_WARNING(
            "[ParseDFSConfig] HCCL_DFS_CONFIG-inconsistent_check was configured to [%s], please configure to "
            "'on' or 'off' or 'first'",
            inconsistentCheckSwitch.c_str());
    }

    std::string taskMonitorInterval = "";
    s32 monitorTime = 0;
    CHK_RET(ParseSingleDFSConfigItem(dfsConfigEnv, TASK_MONITOR_INTERVAL, taskMonitorInterval));
    CHK_RET(ParseMonitor(taskMonitorInterval, monitorTime));

    // 解析连接故障检测时间
    std::string connectionDefaultDetectionTime = "";
    CHK_RET(ParseSingleDFSConfigItem(dfsConfigEnv, CONNECTION_FAULT_DETECTION_TIME, connectionDefaultDetectionTime));
    if (connectionDefaultDetectionTime.empty()) {
        g_envConfig.dfsConnectionFaultDetectionTime = HCCL_MIN_CONNECT_FAULT_DETECTION_TIME;
        HCCL_RUN_INFO(
            "[HCCL_ENV][Parse] HCCL_DFS_CONFIG cluster_heartbeat set by environment to [%d], "
            "stuck_detection set by environment to [%d], connection_fault_detection_time[%d]s "
            "inconsistentCheckSwitch[%d],"
            "task_monitor_interval[%u]ms",
            g_envConfig.enableClusterHeartBeat, g_envConfig.opCounterEnable,
            g_envConfig.dfsConnectionFaultDetectionTime, g_envConfig.inconsistentCheckSwitch,
            g_envConfig.dfsTaskMonitorInterval);
        return HCCL_SUCCESS;
    }
    s32 detctTime = 0;
    HcclResult ret = SalStrToInt(connectionDefaultDetectionTime, HCCL_BASE_DECIMAL, detctTime);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ParseDFSConfig] HCCL_DFS_CONFIG-connection_fault_detection_time[%s]"
            "is invalid, errorno[%d]",
            connectionDefaultDetectionTime.c_str(), ret),
        ret);

    if (detctTime == 0) {
        g_envConfig.dfsConnectionFaultDetectionTime = 0;
    } else if (detctTime >= HCCL_MIN_CONNECT_FAULT_DETECTION_TIME && detctTime <= HCCL_MAX_LINK_TIME_OUT_S) {
        g_envConfig.dfsConnectionFaultDetectionTime = detctTime;
    } else { // 不在允许范围内报错
        HCCL_ERROR(
            "[ParseDFSConfig] HCCL_DFS_CONFIG-connection_fault_detection_time[%d] is invalid, expect: [%d, %d]",
            detctTime, HCCL_MIN_CONNECT_FAULT_DETECTION_TIME, HCCL_MAX_LINK_TIME_OUT_S);
        return HCCL_E_PARA;
    }

    HCCL_RUN_INFO(
        "[HCCL_ENV][Parse] HCCL_DFS_CONFIG cluster_heartbeat set by environment to [%d], "
        "stuck_detection set by environment to [%d], connection_fault_detection_time[%d]s inconsistentCheckSwitch[%d],"
        "task_monitor_interval[%u]ms",
        g_envConfig.enableClusterHeartBeat, g_envConfig.opCounterEnable, g_envConfig.dfsConnectionFaultDetectionTime,
        g_envConfig.inconsistentCheckSwitch, g_envConfig.dfsTaskMonitorInterval);
    return HCCL_SUCCESS;
}

HcclResult GetKeyWordPath(const std::string& cannEnvStr, const std::string& keyStr, std::string& cannPath)
{
    std::string tempPath; // 存放临时路径
    // 查找cann安装路径
    for (u32 i = 0; i < cannEnvStr.length(); ++i) {
        // 环境变量中存放的每段路径之间以':'隔开
        if (cannEnvStr[i] != ':') {
            tempPath += cannEnvStr[i];
        }

        if (cannEnvStr[i] == ':' || i == cannEnvStr.length() - 1) {
            size_t found = tempPath.find(keyStr);
            if (found == std::string::npos) {
                tempPath.clear();
                continue;
            }
            if (tempPath.length() <= found + keyStr.length() || tempPath[found + keyStr.length()] == '/') {
                cannPath = tempPath.substr(0, found + keyStr.length());
                break;
            }
            tempPath.clear();
        }
    }
    if (cannPath.empty()) {
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult ParseLibraryPath(std::string& cannPath)
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_LD_LIBRARY_PATH, mmSysGetEnvValue);
    std::string getPath = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (getPath == "EmptyString") {
        HCCL_ERROR("[ParseLibraryPath]ENV:LD_LIBRARY_PATH is not set");
        return HCCL_E_PARA;
    } else {
        HCCL_INFO("ParseLibraryPath]getPath[%s]", getPath.c_str());
        cannPath = getPath;
    }
    return HCCL_SUCCESS;
}

const bool& GetExternalInputHcclHeartBeatEnable() { return g_envConfig.enableClusterHeartBeat; }

const bool& GetExternalInputStuckDetect() { return g_envConfig.opCounterEnable; }

const InconsistentCheckMode& GetExternalInconsistentCheckSwitch() { return g_envConfig.inconsistentCheckSwitch; }

HcclResult ParseHcclAlgo()
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_ALGO, mmSysGetEnvValue);
    std::string hcclAlgo = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (hcclAlgo != "EmptyString") {
        CHK_RET(SetHcclAlgoConfig(hcclAlgo));
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_ALGO set by environment to [%s]", hcclAlgo.c_str());
    } else {
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_ALGO is not set");
    }
    return HCCL_SUCCESS;
}
