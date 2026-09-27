/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_func.h"
#include <climits>
#include <fstream>
#include <linux/limits.h>
#include <cctype>
#include <algorithm>
#include <sstream>
#include <array>
#include <limits>

#include "string_util.h"
#include "base_config_legacy.h"
#include "orion_adapter_rts.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

static bool ParseDecimalU32(const std::string& value, u32& parsed)
{
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        })) {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long number = std::strtoul(value.c_str(), &end, 10);
    if (errno == ERANGE || end != value.c_str() + value.size() || number > std::numeric_limits<u32>::max()) {
        return false;
    }
    parsed = static_cast<u32>(number);
    return true;
}

bool ParseHostRdmaUdpPorts(const std::string& value, std::vector<std::uint16_t>& ports)
{
    if (value.empty() || value.back() == ',') {
        HCCL_ERROR("[ParseHostRdmaUdpPorts] UDP source port list is empty or ends with a comma.");
        return false;
    }

    std::vector<std::uint16_t> parsedPorts;
    size_t start = 0;
    while (true) {
        const size_t comma = value.find(',', start);
        const std::string token = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        u32 port = 0;
        if (!ParseDecimalU32(token, port) || port == 0 || port > MultiQpSrcPortConfig::CONFIG_SRC_PORT_ID_MAX) {
            HCCL_ERROR(
                "[ParseHostRdmaUdpPorts] UDP source port is invalid, expected a decimal integer in range [1, %u].",
                MultiQpSrcPortConfig::CONFIG_SRC_PORT_ID_MAX);
            return false;
        }
        parsedPorts.emplace_back(static_cast<std::uint16_t>(port));
        if (parsedPorts.size() > MultiQpSrcPortConfig::CONFIG_SRC_PORT_NUM_MAX) {
            HCCL_ERROR(
                "[ParseHostRdmaUdpPorts] UDP source port count[%zu] exceeds the maximum[%u].", parsedPorts.size(),
                MultiQpSrcPortConfig::CONFIG_SRC_PORT_NUM_MAX);
            return false;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1U;
    }
    ports = std::move(parsedPorts);
    return true;
}

/*----------------------------- cast functions -------------------------*/

HostRdmaUdpPortsList CastHostRdmaUdpPortsList(const std::string& s)
{
    HostRdmaUdpPortsList list;
    if (s.empty()) {
        return list;
    }
    if (s.size() > HostRdmaUdpPortsList::CONFIG_VALUE_LEN_MAX) {
        HCCL_ERROR(
            "[CastHostRdmaUdpPortsList] config value length[%zu] exceeds the maximum[%u].", s.size(),
            HostRdmaUdpPortsList::CONFIG_VALUE_LEN_MAX);
        return {};
    }

    size_t start = 0;
    while (start <= s.size()) {
        const size_t semicolon = s.find(';', start);
        const std::string deviceConfig
            = s.substr(start, semicolon == std::string::npos ? std::string::npos : semicolon - start);
        const size_t colon = deviceConfig.find(':');
        if (deviceConfig.empty() || colon == std::string::npos || colon == 0U || colon + 1U >= deviceConfig.size()
            || deviceConfig.find(':', colon + 1U) != std::string::npos) {
            HCCL_ERROR("[CastHostRdmaUdpPortsList] device config has invalid separators or an empty field.");
            return {};
        }

        u32 devicePhyId = 0;
        if (!ParseDecimalU32(deviceConfig.substr(0, colon), devicePhyId)) {
            HCCL_ERROR("[CastHostRdmaUdpPortsList] phy_dev_id is not a valid decimal integer.");
            return {};
        }
        std::vector<std::uint16_t> ports;
        if (!ParseHostRdmaUdpPorts(deviceConfig.substr(colon + 1U), ports)) {
            return {};
        }
        if (!list.portsByPhyId.emplace(devicePhyId, std::move(ports)).second) {
            HCCL_ERROR("[CastHostRdmaUdpPortsList] phy_dev_id[%u] is duplicated.", devicePhyId);
            return {};
        }
        if (semicolon == std::string::npos) {
            break;
        }
        start = semicolon + 1U;
    }
    return list;
}

bool CastBin2Bool(const std::string& s)
{
    bool b = true;
    if (s == "0") {
        b = false;
    } else if (s == "1") {
        b = true;
    } else {
        THROW<InvalidParamsException>(StringFormat("Env config \"%s\" is not valid. Should be 0 or 1", s.c_str()));
    }
    return b;
}

u32 CastBin2UInt(const std::string& s)
{
    u32 b = std::stoi(s);
    if (b > HCCL_CCU_FLAG_NUM) {
        THROW<InvalidParamsException>(StringFormat("Env config \"%s\" is not valid. Should be 0 or 1 or 2", s.c_str()));
    }
    HCCL_INFO("[CastBin2UInt] string[%s] to u32[%u]", s.c_str(), b);
    return b;
}

static HcclResult SplitHcclSocketIfName(const std::string& socketIfName, std::vector<std::string>& configIfNames)
{
    std::size_t start = 0;
    std::size_t end = socketIfName.find(",");
    while (end != std::string::npos) {
        if (start == 0 && end == 0) {
            HCCL_ERROR("[Split][HcclSocketIfName] configIfNames config is invalid.");
            return HCCL_E_PARA;
        }
        configIfNames.push_back(socketIfName.substr(start, end - start));
        start = end + 1;
        end = socketIfName.find(",", start);
    }
    // 处理最后一个部分
    if (start < socketIfName.length()) {
        configIfNames.push_back(socketIfName.substr(start));
    } else if (start == 0) {
        HCCL_ERROR("[Split][HcclSocketIfName] configIfNames config is invalid.");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// 临时方案，且当前未使用，测试期望在此拦截该环境变量所有异常值
SocketIfName CastSocketIfName(const std::string& s)
{
    SocketIfName hcclSocketIfNameGroup{};
    hcclSocketIfNameGroup.configIfNameStr = s;
    std::string hcclSocketIfName = s;
    std::string remainSocketIfName = hcclSocketIfName;
    bool searchNot = false;
    bool searchExact = false;

    if (hcclSocketIfName.length() != 0) {
        // 获取HCCL_SOCKET_IFNAME环境变量匹配规则
        if (!hcclSocketIfName.empty() && hcclSocketIfName.at(0) == '^') {
            searchNot = true;
            // 获取从1位置开始剩余部分环境变量内容
            remainSocketIfName = hcclSocketIfName.substr(1);
        }

        if (!remainSocketIfName.empty() && remainSocketIfName.at(0) == '=') {
            searchExact = true;
            remainSocketIfName = remainSocketIfName.substr(1);
        }

        // 获取用户输入的网卡名列表(使用逗号隔开),将网卡名列表存放到vector变量中
        HcclResult ret = SplitHcclSocketIfName(remainSocketIfName, hcclSocketIfNameGroup.configIfNames);
        if (ret != HCCL_SUCCESS) {
            THROW<InvalidParamsException>(StringFormat(
                "environmental variable HCCL_SOCKET_IFNAME[%s] is invalid. "
                "please check.",
                s.c_str()));
        }
        HCCL_INFO("HCCL_SOCKET_IFNAME set by environment to [%s]", hcclSocketIfName.c_str());
    } else {
        HCCL_INFO("HCCL_SOCKET_IFNAME set by default to [%s]", hcclSocketIfName.c_str());
    }
    hcclSocketIfNameGroup.searchNot = searchNot;
    hcclSocketIfNameGroup.searchExact = searchExact;
    return hcclSocketIfNameGroup;
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

void GetUIntFromStr(const std::string& digitStr, u32& val)
{
    bool isAllDigits = std::all_of(digitStr.begin(), digitStr.end(), ::isdigit);
    CHK_PRT_THROW(
        !isAllDigits, HCCL_ERROR("[GetUIntFromStr] str[%s] is not all digit.", digitStr.c_str()),
        InvalidParamsException, "parser portRange fail.");
    const bool parseSuccess = ParseDecimalU32(digitStr, val);
    CHK_PRT_THROW(
        !parseSuccess, HCCL_ERROR("[GetUIntFromStr] str[%s] is a invalid number.", digitStr.c_str()),
        InvalidParamsException, "parser portRange fail.");
}

void SplitSinglePortRange(const std::string& envName, std::string& rangeStr, SocketPortRange& portRange)
{
    std::string rangeMin{};
    const std::string delim = "-";
    if (SplitString(rangeStr, rangeMin, delim)) {
        GetUIntFromStr(rangeMin, portRange.min);
        GetUIntFromStr(rangeStr, portRange.max);
    } else {
        GetUIntFromStr(rangeStr, portRange.min);
        portRange.max = portRange.min;
    }
    HCCL_INFO(
        "[SplitSinglePortRange] Load hccl socket port range [%u, %u] from %s", portRange.min, portRange.max,
        envName.c_str());
}

void CheckSocketPortRangeValid(const std::string& envName, const std::vector<SocketPortRange>& portRanges)
{
    std::vector<SocketPortRange> rangeVec(portRanges.begin(), portRanges.end());
    std::sort(rangeVec.begin(), rangeVec.end(), [](SocketPortRange& a, SocketPortRange& b) {
        return (a.min == b.min) ? (a.max < b.max) : (a.min < b.min);
    });
    for (size_t i = 0; i < rangeVec.size(); ++i) {
        // the socket range should not be inverted
        CHK_PRT_THROW(
            rangeVec[i].min > rangeVec[i].max,
            HCCL_ERROR(
                "[%s] In %s, in socket port range [%u, %u], the lower bound is greater than"
                " the upper bound.",
                __func__, envName.c_str(), rangeVec[i].min, rangeVec[i].max),
            InvalidParamsException, "check portRange fail.");

        // the socket range should not include the reserved port for auto listening.
        CHK_PRT_THROW(
            (rangeVec[i].min <= HCCL_SOCKET_PORT_RANGE_AUTO),
            HCCL_ERROR(
                "[%s] In %s, socket port range [%u, %u] includes the reserved port number [%u]. "
                "please do not use port [%u] in socket port range.",
                __func__, envName.c_str(), rangeVec[i].min, rangeVec[i].max, HCCL_SOCKET_PORT_RANGE_AUTO,
                HCCL_SOCKET_PORT_RANGE_AUTO),
            InvalidParamsException, "check portRange fail.");

        // the socket range should not exceed the maximum port number
        CHK_PRT_THROW(
            rangeVec[i].max > MAX_PORT_NUMBER,
            HCCL_ERROR(
                "[%s] In %s, in socket port range [%u, %u], the upper bound exceed max port number[%u].", __func__,
                envName.c_str(), rangeVec[i].min, rangeVec[i].max, MAX_PORT_NUMBER),
            InvalidParamsException, "check portRange fail.");

        // the socket range should not be overlapped
        CHK_PRT_THROW(
            i != 0 && rangeVec[i - 1].max >= rangeVec[i].min,
            HCCL_ERROR(
                "[%s] In %s, socket port range [%u, %u] is conflict with socket port range [%u, %u].", __func__,
                envName.c_str(), rangeVec[i - 1].min, rangeVec[i - 1].max, rangeVec[i].min, rangeVec[i].max),
            InvalidParamsException, "check portRange fail.");
    }
}

void SplitHcclSocketPortRange(
    const std::string& envName, std::string& portRangeConfig, std::vector<SocketPortRange>& portRangeVec)
{
    std::string rangeStr{};
    const std::string delim = ",";
    while (SplitString(portRangeConfig, rangeStr, delim)) {
        SocketPortRange portRange = {};
        SplitSinglePortRange(envName, rangeStr, portRange);
        portRangeVec.emplace_back(portRange);
    }
    SocketPortRange portRange = {};
    SplitSinglePortRange(envName, portRangeConfig, portRange);
    portRangeVec.emplace_back(portRange);

    CheckSocketPortRangeValid(envName, portRangeVec);
}

void PrintSocketPortRange(const std::string& envName, const std::vector<SocketPortRange>& portRangeVec)
{
    // assemble port ranges into a string to print the result range
    std::ostringstream portRangeOss;
    for (auto range : portRangeVec) {
        portRangeOss << " [" << std::to_string(range.min) << ", " << std::to_string(range.max) << "]";
    }
    HCCL_INFO("%s is set to%s.", envName.c_str(), portRangeOss.str().c_str());
}

std::vector<SocketPortRange> CastSocketPortRange(const std::string& s, const std::string& envName)
{
    std::vector<SocketPortRange> hcclSocketPortRange;
    // the environment variable is not set
    std::string socketPortRange = s;
    if (socketPortRange.length() == 0) {
        return hcclSocketPortRange;
    }

    // the socket port range is set to auto, then the os will listen on the ports dynamically and automatically.
    if (socketPortRange == HCCL_AUTO_PORT_CONFIG) {
        SocketPortRange autoSocketPortRange = {HCCL_SOCKET_PORT_RANGE_AUTO, HCCL_SOCKET_PORT_RANGE_AUTO};
        hcclSocketPortRange.emplace_back(autoSocketPortRange);

        HCCL_INFO(
            "HCCL_HOST_SOCKET_PORT_RANGE is set to %s as [%u, %u].", HCCL_AUTO_PORT_CONFIG, autoSocketPortRange.min,
            autoSocketPortRange.max);
        return hcclSocketPortRange;
    }

    // the environment variable is set to an empty string
    socketPortRange.erase(std::remove(socketPortRange.begin(), socketPortRange.end(), ' '), socketPortRange.end());
    if (socketPortRange.empty()) {
        return hcclSocketPortRange;
    }

    // load ranges from string
    SplitHcclSocketPortRange(envName, socketPortRange, hcclSocketPortRange);
    CHK_PRT_THROW(
        hcclSocketPortRange.size() == 0,
        HCCL_ERROR("Load empty port range from HCCL_HOST_SOCKET_PORT_RANGE, should not empty, please check."),
        InvalidParamsException, "parser portRange fail.");

    PrintSocketPortRange(envName, hcclSocketPortRange);
    return hcclSocketPortRange;
}

constexpr u32 HCCL_RDMA_TC_BASE = 4; // RDMATrafficClass需要是4的整数倍
void CheckRDMATrafficClass(const u32& rdmaTrafficClass)
{
    if (rdmaTrafficClass % HCCL_RDMA_TC_BASE != 0) {
        RPT_ENV_ERR(
            true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>(
                {std::to_string(rdmaTrafficClass), "HCCL_RDMA_TC", "value should be multiple of four"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rdmaTrafficClass[%u] is not a multiple of [%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_ENV_CONFIG.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), rdmaTrafficClass, HCCL_RDMA_TC_BASE);
        THROW<InvalidParamsException>(
            StringFormat("rdmaTrafficClass[%u] is not a multiple of [%u]", rdmaTrafficClass, HCCL_RDMA_TC_BASE));
    }
}

constexpr u32 BYTES_PER_KB = 1024; // 单位换算
void ConvertUnitQpThreshold(u32& multiQpThreshold)
{
    HCCL_RUN_INFO(
        "[Init][TransferUnitQpThreshold]Env config HCCL_MULTI_QP_THRESHOLD is %u[KB], converted to %u[B] for "
        "subsequent use.",
        multiQpThreshold, multiQpThreshold * BYTES_PER_KB);
    multiQpThreshold *= BYTES_PER_KB;
}

static void ParseAlgoLevel(const std::string& algoLevel, u32& level, HcclAlgoType& algoType)
{
    std::size_t found = algoLevel.find(':');
    if ((found == 0) || (found == (algoLevel.length() - 1))) {
        THROW<InvalidParamsException>("algo config is invalid.");
    }
    if (found == std::string::npos) {
        THROW<InvalidParamsException>("algoLevel cannot find \":\".");
    }

    std::string orginalLevel = algoLevel.substr(0, found);
    std::string orginalAlgo = algoLevel.substr(found + 1);

    const std::map<std::string, u32> hcclAlgoLevelMap
        = {{"level0", HCCL_ALGO_LEVEL_0},
           {"level1", HCCL_ALGO_LEVEL_1},
           {"level2", HCCL_ALGO_LEVEL_2},
           {"level3", HCCL_ALGO_LEVEL_3}};

    const std::map<std::string, HcclAlgoType> hcclAlgoTypeMap = {
        {"null", HcclAlgoType::HCCL_ALGO_TYPE_NULL},
        {"ring", HcclAlgoType::HCCL_ALGO_TYPE_RING},
        {"pipeline", HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE},
        {"fullmesh", HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH},
        {"H-D_R", HcclAlgoType::HCCL_ALGO_TYPE_HDR},
        {"pairwise", HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE},
        {"NHR", HcclAlgoType::HCCL_ALGO_TYPE_NHR},
        {"NB", HcclAlgoType::HCCL_ALGO_TYPE_NB},
        {"NA", HcclAlgoType::HCCL_ALGO_TYPE_NA},
        {"NHR_V1", HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1},
        {"AHC", HcclAlgoType::HCCL_ALGO_TYPE_AHC},
    };

    auto iterAlgoLevel = hcclAlgoLevelMap.find(orginalLevel);
    if (iterAlgoLevel == hcclAlgoLevelMap.end()) {
        THROW<InvalidParamsException>(
            StringFormat("algo config is invalid, level %s is not supported.", orginalLevel.c_str()));
    }

    auto iterAlgoType = hcclAlgoTypeMap.find(orginalAlgo);
    if (iterAlgoType == hcclAlgoTypeMap.end()) {
        THROW<InvalidParamsException>(
            StringFormat("algo config is invalid, algo %s is not supported.", orginalAlgo.c_str()));
    }

    level = iterAlgoLevel->second;
    algoType = iterAlgoType->second;
}

std::vector<HcclAlgoType> CastAlgoTypeVec(const std::string& s)
{
    std::vector<HcclAlgoType> algoTypeVec(HCCL_ALGO_LEVEL_NUM);
    std::string algoConfig = s;
    algoConfig.erase(std::remove(algoConfig.begin(), algoConfig.end(), ' '), algoConfig.end());

    for (u32 i = 0; i < HCCL_ALGO_LEVEL_NUM; i++) {
        algoTypeVec[i] = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    }

    if (algoConfig.empty()) {
        HCCL_RUN_INFO("hccl algo config is empty, HCCL use built-in algo selection.");
        return algoTypeVec;
    }

    std::vector<std::string> algoLevels = SplitString(algoConfig, ';');
    if (algoLevels.size() > HCCL_ALGO_LEVEL_NUM) {
        THROW<InvalidParamsException>(
            StringFormat("The number of algo levels is greater than %u.", HCCL_ALGO_LEVEL_NUM));
    }
    for (const auto& algoLevel : algoLevels) {
        u32 level = 0;
        HcclAlgoType algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        ParseAlgoLevel(algoLevel, level, algo);
        // 检查是否存在重复配置level
        if (algoTypeVec[level] != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            THROW<InvalidParamsException>(
                StringFormat("hccl algo config[%s] is invalid. expect: levelX:algo1;levelY:algo2", algoConfig.c_str()));
        }
        algoTypeVec[level] = algo;
    }

    DevType devType = HrtGetDeviceType(); // 910A3场景只支持level0为ring算法
    if (devType == DevType::DEV_TYPE_910A3 && algoTypeVec[HCCL_ALGO_LEVEL_0] != HcclAlgoType::HCCL_ALGO_TYPE_RING) {
        algoTypeVec[HCCL_ALGO_LEVEL_0] = HcclAlgoType::HCCL_ALGO_TYPE_RING;
    }

    return algoTypeVec;
}

HcclResult SplitHcclOpType(const std::string& algoConfig, std::vector<std::string>& algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find("/");
    if ((found == 0) || (found == (algoConfig.length() - 1))) {
        HCCL_ERROR("[Split][SplitHcclOpType] algo config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainAlgoConfig = algoConfig.substr(found + 1);
    }
    algos.push_back(algoConfig.substr(0, found));
    if (!remainAlgoConfig.empty()) {
        CHK_RET(SplitHcclOpType(remainAlgoConfig, algos));
    }
    return HCCL_SUCCESS;
}

// 新的逐算法的配置和原有的统一配置只可使用一种，发现同时存在时报错
HcclResult CheckAlgoConfigValid(std::vector<std::string>& algos, bool& anyCommonConfig, bool& anySpecificConfig)
{
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        if ((found == 0) || (found == (algConfig.length() - 1))) {
            HCCL_ERROR("[Split][CheckAlgoConfigValid] algo config is invalid.");
            return HCCL_E_PARA;
        } else if (found != std::string::npos) {
            anySpecificConfig = true;
        } else {
            anyCommonConfig = true;
        }
    }
    if (anyCommonConfig && anySpecificConfig) {
        HCCL_ERROR("[CheckAlgoConfigValid]should not set both algo config way");
        return HCCL_E_PARA;
    }
    if (anyCommonConfig && algos.size() > 1) {
        HCCL_ERROR("[CheckAlgoConfigValid]should only set one common config");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ParserHcclAlgoLevel(const std::string& algoLevel, u32& level, HcclAlgoType& algoType)
{
    std::size_t found = algoLevel.find(":");
    if ((found == 0) || (found == (algoLevel.length() - 1))) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    }
    if (found == std::string::npos) {
        THROW<InvalidParamsException>("algoLevel cannot find \":\".");
    }

    std::string orginalLevel = algoLevel.substr(0, found);
    std::string orginalAlgo = algoLevel.substr(found + 1);

    const std::map<std::string, u32> hcclAlgoLevelMap
        = {{"level0", HCCL_ALGO_LEVEL_0},
           {"level1", HCCL_ALGO_LEVEL_1},
           {"level2", HCCL_ALGO_LEVEL_2},
           {"level3", HCCL_ALGO_LEVEL_3}};

    const std::map<std::string, HcclAlgoType> hcclAlgoTypeMap = {
        {"null", HcclAlgoType::HCCL_ALGO_TYPE_NULL},
        {"ring", HcclAlgoType::HCCL_ALGO_TYPE_RING},
        {"pipeline", HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE},
        {"fullmesh", HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH},
        {"H-D_R", HcclAlgoType::HCCL_ALGO_TYPE_HDR},
        {"pairwise", HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE},
        {"NHR", HcclAlgoType::HCCL_ALGO_TYPE_NHR},
        {"NB", HcclAlgoType::HCCL_ALGO_TYPE_NB},
        {"NA", HcclAlgoType::HCCL_ALGO_TYPE_NA},
        {"NHR_V1", HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1},
        {"AHC", HcclAlgoType::HCCL_ALGO_TYPE_AHC},
    };

    auto iterAlgoLevel = hcclAlgoLevelMap.find(orginalLevel);
    if (iterAlgoLevel == hcclAlgoLevelMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, level %s is not supported.", orginalLevel.c_str());
        return HCCL_E_PARA;
    }

    auto iterAlgoType = hcclAlgoTypeMap.find(orginalAlgo);
    if (iterAlgoType == hcclAlgoTypeMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, algo %s is not supported.", orginalAlgo.c_str());
        return HCCL_E_PARA;
    }

    level = iterAlgoLevel->second;
    algoType = iterAlgoType->second;

    return HCCL_SUCCESS;
}

const std::map<HcclAlgoType, std::string> HcclAlgoTypeMap = {
    {HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, "default"},   {HcclAlgoType::HCCL_ALGO_TYPE_RING, "ring"},
    {HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE, "pipeline"}, {HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, "fullmesh"},
    {HcclAlgoType::HCCL_ALGO_TYPE_HDR, "HDR"},           {HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE, "pairwise"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NHR, "NHR"},           {HcclAlgoType::HCCL_ALGO_TYPE_NB, "NB"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NULL, "null"},         {HcclAlgoType::HCCL_ALGO_TYPE_NA, "NA"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1, "NHR_V1"},     {HcclAlgoType::HCCL_ALGO_TYPE_AHC, "AHC"},
};

HcclResult SplitHcclAlgoLevel(const std::string& algoConfig, std::vector<std::string>& algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find(";");
    if ((found == 0) || (found == (algoConfig.length() - 1))) {
        HCCL_ERROR("[Split][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainAlgoConfig = algoConfig.substr(found + 1);
    } else {
        // 最后一组配置,剩余的字符串为空
    }
    algos.push_back(algoConfig.substr(0, found));

    if (algos.size() > HCCL_ALGO_LEVEL_NUM) {
        HCCL_ERROR("[Split][HcclAlgoLevel] algo config is invalid. algo level is more than %u.", HCCL_ALGO_LEVEL_NUM);
        return HCCL_E_PARA;
    }
    if (!remainAlgoConfig.empty()) {
        CHK_RET(SplitHcclAlgoLevel(remainAlgoConfig, algos));
    }

    return HCCL_SUCCESS;
}

HcclResult ParseAlgoString(std::string opName, std::string& algoString, std::vector<HcclAlgoType>& algType)
{
    algType = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    std::vector<std::string> algoLevels;
    HcclResult ret = SplitHcclAlgoLevel(algoString, algoLevels);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
            "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
            algoString.c_str()),
        ret);
    for (auto algoLevel : algoLevels) {
        u32 level = 0;
        HcclAlgoType algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        ret = ParserHcclAlgoLevel(algoLevel, level, algo);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
                "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
                algoString.c_str()),
            ret);
        // 检查是否存在重复配置level
        if (algType[level] != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            HCCL_ERROR(
                "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
                "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
                algoString.c_str());
            return HCCL_E_PARA;
        }
        algType[level] = algo;
    }
    auto level0Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_0]);
    auto level1Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_1]);
    auto level2Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_2]);
    auto level3Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_3]);
    HCCL_RUN_INFO(
        "hccl algo op %s config: level0:%s, level1:%s, level2:%s, level3:%s", opName.c_str(),
        level0Iter->second.c_str(), level1Iter->second.c_str(), level2Iter->second.c_str(), level3Iter->second.c_str());
    return HCCL_SUCCESS;
}

HcclResult
SetCommonAlgType(std::vector<std::string>& algos, std::map<OpType, std::vector<HcclAlgoType>>& hcclAlgoConfig)
{
    std::vector<HcclAlgoType> algType;
    CHK_RET(ParseAlgoString("all op type", algos[0], algType));
    for (auto opType : OP_TYPE_SET) {
        hcclAlgoConfig[opType] = algType;
    }
    return HCCL_SUCCESS;
}

HcclResult
SetSpecificAlgType(std::vector<std::string>& algos, std::map<OpType, std::vector<HcclAlgoType>>& hcclAlgoConfig)
{
    std::map<std::string, OpType> hcclOpTypeMap = {
        {"broadcast", OpType::BROADCAST},
        {"allreduce", OpType::ALLREDUCE},
        {"reduce", OpType::REDUCE},
        {"send", OpType::SEND},
        {"receive", OpType::RECV},
        {"allgather", OpType::ALLGATHER},
        {"reducescatter", OpType::REDUCESCATTER},
        {"alltoall", OpType::ALLTOALL},
        {"gather", OpType::GATHER},
        {"scatter", OpType::SCATTER},
        {"sendrecv", OpType::BATCHSENDRECV},
    };
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        std::string opStringName = algConfig.substr(0, found);
        if (hcclOpTypeMap.find(opStringName) != hcclOpTypeMap.end()) {
            OpType optype = hcclOpTypeMap[opStringName];
            std::string remainAlgoConfig = algConfig.substr(found + 1);
            std::vector<HcclAlgoType> algType;
            CHK_RET(ParseAlgoString(opStringName, remainAlgoConfig, algType));
            if (algType[0] == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
                HCCL_ERROR("[SetSpecificAlgType] specific config level0 not support null type.");
                return HCCL_E_PARA;
            }
            hcclAlgoConfig[optype] = algType;
        } else {
            HCCL_ERROR(
                "[SetSpecificAlgType] specific config optype[%s] is invalid, please check", opStringName.c_str());
            return HCCL_E_PARA;
        }
    }
    if (hcclAlgoConfig.find(OpType::ALLTOALL) != hcclAlgoConfig.end()) {
        hcclAlgoConfig[OpType::ALLTOALLV] = hcclAlgoConfig[OpType::ALLTOALL];
        hcclAlgoConfig[OpType::ALLTOALLVC] = hcclAlgoConfig[OpType::ALLTOALL];
    }
    return HCCL_SUCCESS;
}

std::map<OpType, std::vector<HcclAlgoType>> SetHcclAlgoConfig(const std::string& hcclAlgo)
{
    std::string algoConfig = hcclAlgo;
    algoConfig.erase(std::remove(algoConfig.begin(), algoConfig.end(), ' '), algoConfig.end());
    std::map<OpType, std::vector<HcclAlgoType>> hcclAlgoConfig;
    if (algoConfig.empty()) {
        HCCL_RUN_INFO("hccl algo config is empty, HCCL use built-in algo selection.");
        return hcclAlgoConfig;
    }
    std::vector<std::string> algoPerOptype;
    HcclResult splitRet = SplitHcclOpType(algoConfig, algoPerOptype);
    if (splitRet != HCCL_SUCCESS) {
        THROW<InvalidParamsException>(StringFormat(
            "Env HCCL_ALGO config \"%s\" is invalid. example [level0:NA;level1:NHR] or"
            "[allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R]",
            hcclAlgo.c_str()));
    }

    bool anyCommonConfig = false;
    bool anySpecificConfig = false;
    HcclResult checkRet = CheckAlgoConfigValid(algoPerOptype, anyCommonConfig, anySpecificConfig);
    if (checkRet != HCCL_SUCCESS) {
        THROW<InvalidParamsException>(StringFormat(
            "Env HCCL_ALGO config \"%s\" is invalid. example [level0:NA;level1:NHR] or"
            "[allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R]",
            hcclAlgo.c_str()));
    }
    HcclResult ret = HCCL_SUCCESS;
    if (anyCommonConfig) {
        ret = SetCommonAlgType(algoPerOptype, hcclAlgoConfig);
    } else {
        ret = SetSpecificAlgType(algoPerOptype, hcclAlgoConfig);
    }
    if (ret != HCCL_SUCCESS) {
        THROW<InvalidParamsException>(StringFormat(
            "Env HCCL_ALGO config \"%s\" is invalid. example [level0:NA;level1:NHR] or"
            "[allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R]",
            hcclAlgo.c_str()));
    }
    return hcclAlgoConfig;
}

HcclAccelerator CastHcclAccelerator(const std::string& s)
{
    HcclAccelerator mode;
    if (s == "AI_CPU" || s == "AICPU_TS" || s == "AICPU_CacheDisable") {
        mode = HcclAccelerator::AICPU_TS;
    } else if (s == "AIV") {
        mode = HcclAccelerator::AIV;
    } else if (s == "CCU_MS") {
        mode = HcclAccelerator::CCU_MS;
    } else if (s == "CCU_SCHED") {
        mode = HcclAccelerator::CCU_SCHED;
    } else {
        HCCL_ERROR(
            "Env HCCL_OP_EXPANSION_MODE config do not support %s, it should be one of [AI_CPU, AICPU_TS, "
            "AICPU_CacheDisable, AIV, CCU_MS, CCU_SCHED].",
            s.c_str());
        THROW<InvalidParamsException>(StringFormat(
            "Env HCCL_OP_EXPANSION_MODE config \"%s\" is invalid."
            "it should be one of [AI_CPU, AICPU_TS, AICPU_CacheDisable, AIV, CCU_MS, CCU_SCHED].",
            s.c_str()));
    }
    return mode;
}

s32 CastSocketFamily(const std::string& s)
{
    s32 hcclSocketFamily;
    if (s == "AF_INET") {
        hcclSocketFamily = AF_INET;
    } else if (s == "AF_INET6") {
        hcclSocketFamily = AF_INET6;
    } else {
        hcclSocketFamily = -1;
        THROW<InvalidParamsException>(StringFormat(
            "environmental variable HCCL_SOCKET_FAMILY[%s] is invalid. it should "
            "be \"AF_INET\" or \"AF_INET6\".",
            s.c_str()));
    }
    return hcclSocketFamily;
}

std::string GetCannVersionPath(const std::string& cannEnvStr, const std::string& keyStr)
{
    std::string cannVersionPath;
    std::string tempPath; // 存放临时路径
    // 查找cann安装路径
    for (u32 i = 0; i < cannEnvStr.length(); ++i) {
        // 环境变量中存放的每段路径之间以':'隔开
        if (cannEnvStr[i] != ':') {
            tempPath += cannEnvStr[i];
        }
        // 对存放CANN版本文件的路径进行搜索, 有两种情况
        // 一种是*/latest/version.cfg
        // 另一种是*/runtime/version.info
        if (cannEnvStr[i] == ':' || i == cannEnvStr.length() - 1) {
            size_t found = tempPath.find(keyStr);
            if (found == string::npos) {
                tempPath.clear();
                continue;
            }
            // 防止出现类似/runtime*/的情况
            if (tempPath.length() <= found + keyStr.length() || tempPath[found + keyStr.length()] == '/') {
                cannVersionPath = tempPath.substr(0, found + keyStr.length());
                break;
            }
            tempPath.clear();
        }
    }
    // 路径为空
    if (cannVersionPath.empty()) {
        return "NotFound";
    }
    return cannVersionPath;
}

std::string LoadCannVersionInfoFile(const std::string& realName, const std::string& keyStr)
{
    std::string cannVersion;
    // 打开该文件前，判断该文件路径是否有效、规范
    char realFile[PATH_MAX] = {0};
    if (realpath(realName.c_str(), realFile) == nullptr) {
        HCCL_INFO("[CannVersion][Verification]cann version path %s is not a valid real path", realName.c_str());
        return "";
    }
    HCCL_INFO("Load CannVersion InfoFile in %s", realFile);

    // realFile转str,然后open这个str
    std::ifstream infile(realFile, std::ifstream::in);

    if (!infile.is_open()) {
        HCCL_INFO("[CannVersion][Verification]%s does not exist.", realFile);
        return "";
    }

    // 逐行读取，结果放在line中，寻找带有keyStr的字符串
    string line;
    s32 maxRows = 100; // 在文件中读取的最长行数为100，避免超大文件长时间读取
    while (getline(infile, line)) {
        --maxRows;
        if (maxRows < 0) {
            HCCL_WARNING("[CannVersion][Verification]version file content is too long.");
            return "";
        }
        u32 found = line.find(keyStr);
        // 版本字段的两种模式
        // runtime目录下, version.info文件, Version=1.83.T8.0.B128
        // latest目录下, version.cfg文件, runtime_running_version=[1.83.T8.0.B128:CANN-1.83]
        if (found == 0) {
            u32 startPos = keyStr.length();                  // 版本字符串开始位置
            u32 endPos = min(line.find(":"), line.length()); // 版本字符串在":"或结尾处结束
            // 版本字符串为空
            if (endPos <= startPos) {
                HCCL_WARNING("[CannVersion][Verification]cannVersion is invalid.");
                return "";
            }

            u32 len = endPos - startPos;              // 版本字符串长度
            cannVersion = line.substr(startPos, len); // 从keyStr截断
            HCCL_INFO("[Parse][CannVersion]success, CannVersion is %s ", cannVersion.c_str());
            break;
        }
    }
    infile.close();
    return cannVersion;
}

std::string CastCannVersion(const std::string& cannEnv)
{
    std::string cannVersionPath = GetCannVersionPath(cannEnv, "/runtime");
    if (cannVersionPath != "NotFound") {
        cannVersionPath += "/version.info";
        std::string cannVersion = LoadCannVersionInfoFile(cannVersionPath, "Version=");
        return cannVersion;
    }

    cannVersionPath = GetCannVersionPath(cannEnv, "/latest");
    if (cannVersionPath != "NotFound") {
        cannVersionPath += "/version.cfg";
        std::string cannVersion = LoadCannVersionInfoFile(cannVersionPath, "runtime_running_version=[");
        return cannVersion;
    }

    HCCL_INFO("cannot find the version file in %s.", cannEnv.c_str());
    return "";
}

std::vector<std::string> SplitDfsConfig(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::istringstream stream(str);
    std::string token;

    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    if (stream.peek() != EOF) {
        std::string remaining;
        std::getline(stream, remaining);
        tokens.push_back(remaining);
    }
    if (!str.empty() && str.back() == delimiter) {
        tokens.push_back("");
    }
    return tokens;
}

void CastDfsConfigParseTaskExceptionEnable(bool& taskExceptionEnable, const std::string configValue)
{
    if (configValue == "off") {
        taskExceptionEnable = false;
        HCCL_WARNING("env[HCCL_DFS_CONFIG] task_exception was configured to [%s]", configValue.c_str());
    } else if (configValue == "on") {
        taskExceptionEnable = true;
    } else {
        THROW<InvalidParamsException>(
            StringFormat("env[HCCL_DFS_CONFIG] please set task_exception to 'on' or 'off'.", configValue.c_str()));
    }
}

void CastDfsConfigParseClusterHeartBeatEnable(bool& clusterHeartBeatEnable, const std::string configValue)
{
    if (configValue == "off") {
        clusterHeartBeatEnable = false;
        HCCL_WARNING("env[HCCL_DFS_CONFIG] cluster_heartbeat was configured to [%s]", configValue.c_str());
    } else if (configValue == "on") {
        clusterHeartBeatEnable = true;
    } else {
        THROW<InvalidParamsException>(
            StringFormat("env[HCCL_DFS_CONFIG] please set cluster_heartbeat to 'on' or 'off'.", configValue.c_str()));
    }
}

void CastDfsConfigParseRankConsistentState(int32_t& rankConsistentState, const std::string configValue)
{
    if (configValue == "off") {
        rankConsistentState = -1;
    } else if (configValue == "first") {
        rankConsistentState = 0;
    } else if (configValue == "on") {
        rankConsistentState = 1;
    } else {
        THROW<InvalidParamsException>(StringFormat(
            "env[HCCL_DFS_CONFIG] please set inconsistent_check to '-1' or '0' or '1'.", configValue.c_str()));
    }
}

DfsConfig CastDfsConfig(const std::string& dfsConfigEnv)
{
    constexpr std::size_t DFS_CONFIG_ITEM_NUM = 3;
    const std::array<std::string, DFS_CONFIG_ITEM_NUM> dfsItemName
        = {"task_exception", "cluster_heartbeat", "inconsistent_check"};
    bool taskExceptionEnable = true;
    bool clusterHeartBeatEnable = true;
    int32_t rankConsistentState = 0;
    std::string dfsConfigEnvCopy = dfsConfigEnv;
    dfsConfigEnvCopy.erase(std::remove(dfsConfigEnvCopy.begin(), dfsConfigEnvCopy.end(), ' '), dfsConfigEnvCopy.end());
    auto items = SplitDfsConfig(dfsConfigEnvCopy, ',');
    for (const auto& item : items) {
        auto itemPair = SplitDfsConfig(item, ':');
        constexpr std::size_t ITEM_SIZE = 2;
        if (itemPair.size() != ITEM_SIZE
            || std::find(dfsItemName.begin(), dfsItemName.end(), itemPair[0]) == dfsItemName.end()) {
            THROW<InvalidParamsException>(StringFormat(
                "env[HCCL_DFS_CONFIG] value[%s] is invalid, please check, example [task_exception:on]",
                dfsConfigEnv.c_str()));
        }
        if (itemPair[0] == dfsItemName[0]) {
            CastDfsConfigParseTaskExceptionEnable(taskExceptionEnable, itemPair[1]);
        } else if (itemPair[0] == dfsItemName[1]) {
            CastDfsConfigParseClusterHeartBeatEnable(clusterHeartBeatEnable, itemPair[1]);
        } else if (itemPair[0] == dfsItemName[2]) {
            CastDfsConfigParseRankConsistentState(rankConsistentState, itemPair[1]);
        }
    }
    DfsConfig config{taskExceptionEnable, clusterHeartBeatEnable, rankConsistentState};

    return config;
}

/*----------------------------- validate functions -------------------------*/
void CheckExecTimeOut(const u32& timeOut)
{
    DevType devType = HrtGetDeviceType();
    if (devType == DevType::DEV_TYPE_910A2 || devType == DevType::DEV_TYPE_910A3 || devType == DevType::DEV_TYPE_950
        || devType == DevType::DEV_TYPE_960) {
        // 910A2和910A3算子超时时间范围0s-2147483647s,其中0代表永不超时
        CheckRange<u32>(timeOut, 0, HCCL_EXEC_TIME_OUT_S_910A3);
    } else {
        // 非910A2和910A3算子超时时间范围1s-17340s
        CheckRange<u32>(timeOut, 1, HCCL_EXEC_TIME_OUT_S);
    }
}

void CheckFilePath(const string& filePath)
{
    if (filePath.length() >= (PATH_MAX) || filePath.length() == 0) {
        THROW<InvalidParamsException>(StringFormat(
            "env[HCCL_WHITELIST_FILE] or env[HCCL_RDMA_QP_PORT_CONFIG_PATH] is invalid, len is %u, should be (0,4096)",
            filePath.length()));
    }
}

void CheckRdmaTimeout(const u32& timeout)
{
    // HCCL_RDMA_TIMEOUT 合法范围为：[0, 31]，超出31按照0进行处理，表示永不超时
    // 此处为空实现，原因为非法值在 Str2T<u32> 里面已经处理
    (void)timeout;
}

/*-------------------------- post process functions -------------------------*/
void SetRealPath(string& filePath)
{
    char realFile[PATH_MAX] = {0};
    if (realpath(filePath.c_str(), realFile) == nullptr) {
        THROW<InvalidParamsException>(
            StringFormat("[Init][EnvVarParam]path %s is not a valid real path", filePath.c_str()));
    }
    filePath = std::string(realFile);
}

void ProcExecTimeOut(u32& timeOut)
{
    DevType devType = HrtGetDeviceType();
    if (devType == DevType::DEV_TYPE_910A2 || devType == DevType::DEV_TYPE_910A3 || devType == DevType::DEV_TYPE_950
        || devType == DevType::DEV_TYPE_960) {
        return;
    }
    // 910A芯片限制超时时长为68的倍数
    s32 intPart = timeOut / HCCL_INTEVAL_EXEC_TIME_OUT_S;
    intPart = (intPart == 0) ? 1 : intPart;
    timeOut = intPart * HCCL_INTEVAL_EXEC_TIME_OUT_S;
}

void ProcRdmaTimeout(u32& timeout)
{
    // HCCL_RDMA_TIMEOUT 合法范围为：[0, 31]，超出31按照0进行处理，表示永不超时
    if (timeout > EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MAX) {
        HCCL_WARNING(
            "HCCL_RDMA_TIMEOUT[%u] exceeds max [%u], reset to [%u]", timeout, EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MAX,
            EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MIN);
        timeout = EnvRdmaConfig::HCCL_RDMA_TIMEOUT_MIN;
    }
}

/*-------------------------- detour type -------------------------*/
// 临时方案，特定场景执行算法会报错，后续适配了再放开
HcclDetourType CastDetourType(const std::string& s)
{
    if (s == "detour:1") {
        HCCL_INFO("HCCL detour type is 2P (detour:1).");
        return HcclDetourType::HCCL_DETOUR_ENABLE_2P;
    } else if (s == "detour:0") {
        HCCL_INFO("HCCL detour type is disable (detour:0).");
    } else {
        THROW<NotSupportException>(StringFormat("environment variable HCCL_DETOUR currently only supports"
                                                " detour:1 and detour:0 or not set."));
    }
    return HcclDetourType::HCCL_DETOUR_DISABLE;
}

/*----------------------- multi qp src port config --------------------------*/
std::vector<std::uint16_t> GetHostRdmaUdpPortsByPhyId(const HostRdmaUdpPortsList& list, u32 devicePhyId)
{
    const auto iter = list.portsByPhyId.find(devicePhyId);
    if (iter == list.portsByPhyId.end()) {
        return {};
    }
    return iter->second;
}

std::vector<std::uint16_t>
GetMultiQpSrcPortsByIpPair(const MultiQpSrcPortConfig& config, const IpAddress& srcIp, const IpAddress& dstIp)
{
    if (!config.IsAvailable()) {
        return {};
    }

    std::string srcIpStr = srcIp.GetIpStr();
    std::string dstIpStr = dstIp.GetIpStr();
    bool isIpv4 = (srcIp.GetFamily() == AF_INET);

    std::string key = srcIpStr + "," + dstIpStr;
    auto it = config.ipPairToPorts.find(key);
    if (it != config.ipPairToPorts.end()) {
        return it->second;
    }

    key = (isIpv4 ? "0.0.0.0," : "::,") + dstIpStr;
    it = config.ipPairToPorts.find(key);
    if (it != config.ipPairToPorts.end()) {
        return it->second;
    }

    key = srcIpStr + (isIpv4 ? ",0.0.0.0" : ",::");
    it = config.ipPairToPorts.find(key);
    if (it != config.ipPairToPorts.end()) {
        return it->second;
    }

    key = isIpv4 ? "0.0.0.0,0.0.0.0" : "::,::";
    it = config.ipPairToPorts.find(key);
    if (it != config.ipPairToPorts.end()) {
        return it->second;
    }

    return {};
}

u32 GetMultiQpPortsNumByIpPair(const MultiQpSrcPortConfig& config, const IpAddress& srcIp, const IpAddress& dstIp)
{
    auto ports = GetMultiQpSrcPortsByIpPair(config, srcIp, dstIp);
    return static_cast<u32>(ports.size());
}

} // namespace Hccl
