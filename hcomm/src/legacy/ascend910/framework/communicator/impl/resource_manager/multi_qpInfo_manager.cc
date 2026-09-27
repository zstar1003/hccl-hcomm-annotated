/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_qpInfo_manager.h"
#include "adapter_hccp_common.h"
#include <queue>
#include <cstdlib>
#include <fstream>
#include "externalinput_pub.h"
#include "adapter_error_manager_pub.h"
#include "../../../nslbdp/hccl_nslbdp.h"
#include "mmpa_api.h"
namespace hccl {
static std::vector<std::string> devCfgPortMode{"multi_qp", "nslb_dp", ""};

static std::string MultiQpFromToString(MUL_QP_FROM value)
{
    static const std::map<MUL_QP_FROM, std::string> map
        = {{MUL_QP_FROM::MUL_QP_FROM_DEV_CFG, "DEV_CFG"},
           {MUL_QP_FROM::MUL_QP_FROM_DEV_NSLB, "DEV_NSLB"},
           {MUL_QP_FROM::MUL_QP_FROM_ENV_PORT_CONFIG_PATH, "ENV_HCCL_RDMA_QP_PORT_CONFIG_PATH"},
           {MUL_QP_FROM::MUL_QP_FROM_ENV_PER_CONNECTION, "ENV_HCCL_RDMA_QPS_PER_CONNECTION"},
           {MUL_QP_FROM::MUL_QP_FROM_UNKNOWN, "MUL_QP_FROM_UNKNOWN"}};

    auto it = map.find(value);
    if (it != map.end()) {
        return it->second;
    }
    return "UNKNOWN";
}

HcclResult MulQpInfoCacheBase::Init() { return initStatus_; }

void MulQpInfoCacheBase::SetMulQpInfoFrom(MUL_QP_FROM mulQpInfoFrom) { mulQpInfoFrom_ = mulQpInfoFrom; }

MUL_QP_FROM MulQpInfoCacheBase::MulQpInfoFrom() const { return mulQpInfoFrom_; }

HcclResult MulQpInfoCacheBase::GetPortsNumByIpPair(PortNum& portNum, const KeyPair& ipPair) const
{
    (void)portNum;
    (void)ipPair;
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult MulQpInfoCacheBase::GetSpecialSourcePortsByIpPair(MulQpSourcePorts& sourcePorts, const KeyPair& ipPair) const
{
    (void)sourcePorts;
    (void)ipPair;
    return HcclResult::HCCL_E_INTERNAL;
}

DevCfgMulQpInfoCache::DevCfgMulQpInfoCache(const NICDeployment nicDeployment, const std::int32_t phyId)
    : nicDeployment_(nicDeployment),
      phyId_(phyId)
{
    SetMulQpInfoFrom(MUL_QP_FROM::MUL_QP_FROM_DEV_CFG);
}

HcclResult MulQpInfo::Init(const InitParams& params)
{
    std::lock_guard<std::mutex> initLock(initLock_);
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        || (params.GetDevType() != DevType::DEV_TYPE_910_93 && params.GetDevType() != DevType::DEV_TYPE_910B)) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (initStatus_ != HcclResult::HCCL_E_RESERVED) {
        return initStatus_;
    }
    // 生成队列排序
    std::queue<std::unique_ptr<MulQpInfoCacheBase>> parseOrderQueue;
    std::unique_ptr<DevCfgMulQpInfoCache> devCfgMulQpInfoCache;
    std::unique_ptr<DevNslbMulQpInfoCache> devNslbMulQpInfoCache;
    std::unique_ptr<EnvConfigPathCache> envConfigPathCache;
    std::unique_ptr<EnvPerConnectionQpInfoCache> envPerConnectionQpInfoCache;
    EXCEPTION_CATCH(
        (devCfgMulQpInfoCache = std::make_unique<DevCfgMulQpInfoCache>(params.GetNicDeployment(), params.GetPhyId())),
        return HCCL_E_INTERNAL);
    EXCEPTION_CATCH(
        (devNslbMulQpInfoCache = std::make_unique<DevNslbMulQpInfoCache>(params.GetNicDeployment(), params.GetPhyId())),
        return HCCL_E_INTERNAL);
    EXCEPTION_CATCH((envConfigPathCache = std::make_unique<EnvConfigPathCache>()), return HCCL_E_INTERNAL);
    EXCEPTION_CATCH(
        (envPerConnectionQpInfoCache = std::make_unique<EnvPerConnectionQpInfoCache>()), return HCCL_E_INTERNAL);
    parseOrderQueue.emplace(std::move(devCfgMulQpInfoCache));
    parseOrderQueue.emplace(std::move(devNslbMulQpInfoCache));
    parseOrderQueue.emplace(std::move(envConfigPathCache));
    parseOrderQueue.emplace(std::move(envPerConnectionQpInfoCache));
    while (!parseOrderQueue.empty()) {
        auto&& item = parseOrderQueue.front();
        initStatus_ = item->Init();
        if (initStatus_ != HcclResult::HCCL_SUCCESS) {
            return initStatus_;
        }
        if (item->IsAvailable()) {
            config_ = std::move(item);
            break; // 解析成功
        }
        parseOrderQueue.pop();
    }
    if (config_ && config_->IsAvailable()) {
        HCCL_RUN_INFO(
            "[MultiQp][MulQpInfo] Init Success,Device PhyId[%d] MultiQp config from type[%s]", params.GetPhyId(),
            MultiQpFromToString(config_->MulQpInfoFrom()).c_str());
    }
    initStatus_ = HcclResult::HCCL_SUCCESS; // 均未设置 或某种方式解析成功
    return initStatus_;
}

bool MulQpInfo::IsInitialized()
{
    std::lock_guard<std::mutex> initLock(initLock_);
    return initStatus_ != HcclResult::HCCL_E_RESERVED;
}

HcclResult MulQpInfo::IsEnableMulQp(bool& isEnableMulQp)
{
    std::lock_guard<std::mutex> initLock(initLock_);
    isEnableMulQp = initStatus_ == HcclResult::HCCL_SUCCESS && config_ && config_->IsAvailable();
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult DevStrToUint16(const std::string& value, std::uint16_t& count)
{
    HcclResult result = HcclResult::HCCL_SUCCESS;
    unsigned long tempCount = 0;
    try {
        if (!value.empty()) {
            std::size_t parsePos = 0;
            tempCount = std::stoul(value, &parsePos, 0);
            if (parsePos != value.size()) {
                HCCL_ERROR(
                    "[MulQpInfo][StrToUint16]The string is not a valid unsigned integer, str[%s] pos[%llu]"
                    " str size[%llu], val[%lu]",
                    value.c_str(), static_cast<std::uint64_t>(parsePos), static_cast<std::uint64_t>(value.size()),
                    tempCount);
                result = HcclResult::HCCL_E_PARA;
            } else if (tempCount <= 0xFFFFU) {
                count = static_cast<std::uint16_t>(tempCount);
            } else {
                HCCL_ERROR(
                    "[MulQpInfo][StrToUint16]result of stoul is greater than 0xFFFFU, str[%s] base[%d] val[%lu]",
                    value.c_str(), 0, tempCount);
                result = HcclResult::HCCL_E_PARA;
            }
        }
    } catch (std::invalid_argument& e) {
        HCCL_ERROR(
            "[MulQpInfo][StrToUint16]stoul invalid arg: %s, str[%s] base[%d] val[%lu]", e.what(), value.c_str(), 0,
            tempCount);
        result = HcclResult::HCCL_E_PARA;
    } catch (std::out_of_range& e) {
        HCCL_ERROR(
            "[MulQpInfo][StrToUint16]stoul out of range: %s, str[%s] base[%d] val[%lu]", e.what(), value.c_str(), 0,
            tempCount);
        result = HcclResult::HCCL_E_PARA;
    } catch (...) {
        HCCL_ERROR("[MulQpInfo][StrToUint16]stoul catch error, str[%s] base[%d] val[%lu]", value.c_str(), 0, tempCount);
        result = HcclResult::HCCL_E_PARA;
    }
    return result;
};

static HcclResult DevMulPorts(const std::string& value, std::vector<std::uint16_t>& ports)
{
    HcclResult ret = HcclResult::HCCL_SUCCESS;
    if (!value.empty()) {
        std::uint16_t tempCount = 0;
        constexpr char separator = ',';
        std::size_t pos{0U};
        std::size_t start{0U};
        while ((pos = value.find(separator, start)) != std::string::npos) {
            if (pos == start) {
                HCCL_ERROR("[MulQpInfo][DevMulPorts]format error, value[%s]", value.c_str());
                ret = HcclResult::HCCL_E_PARA;
                break;
            }
            ret = DevStrToUint16(value.substr(start, pos - start), tempCount);
            if (ret != HcclResult::HCCL_SUCCESS) {
                return ret;
            }
            ports.emplace_back(tempCount);
            start = pos + 1U;
        }
        if (start >= value.size()) {
            HCCL_ERROR("[MulQpInfo][DevMulPorts]format error, value[%s]", value.c_str());
            ret = HcclResult::HCCL_E_NETWORK;
        }
        (void)ports.emplace_back(std::stoul(value.substr(start), nullptr, 0));
    }
    return ret;
}

HcclResult DevCfgMulQpInfoCache::Init()
{
    std::string modeValue;
    initStatus_ = HrtRaGetHccnCfg(
        static_cast<std::uint32_t>(nicDeployment_), phyId_, HccnCfgKeyT::HCCN_UDP_PORT_MODE, modeValue);
    if (initStatus_ != HcclResult::HCCL_SUCCESS) {
        return initStatus_;
    }
    std::string countValue;
    initStatus_ = HrtRaGetHccnCfg(
        static_cast<std::uint32_t>(nicDeployment_), phyId_, HccnCfgKeyT::HCCN_MULTI_QP_COUNT, countValue);
    if (initStatus_ != HcclResult::HCCL_SUCCESS) {
        return initStatus_;
    }
    std::uint16_t qpCount{0U};
    initStatus_ = DevStrToUint16(countValue, qpCount);
    if (initStatus_ != HcclResult::HCCL_SUCCESS) {
        return initStatus_;
    }
    std::string portValue;
    std::vector<std::uint16_t> qpPorts{};
    initStatus_ = HrtRaGetHccnCfg(
        static_cast<std::uint32_t>(nicDeployment_), phyId_, HccnCfgKeyT::HCCN_MULTI_QP_UDP_PORTS, portValue);
    if (initStatus_ != HcclResult::HCCL_SUCCESS) {
        return initStatus_;
    }
    initStatus_ = DevMulPorts(portValue, qpPorts);
    if (initStatus_ != HcclResult::HCCL_SUCCESS) {
        return initStatus_;
    }
    const bool isNotConfig = modeValue.empty() && countValue.empty() && portValue.empty();
    const bool isNumQpConfigSuccess
        = !modeValue.empty() && modeValue == "multi_qp" && qpCount >= 1 && qpPorts.size() == qpCount;
    constexpr std::size_t afterPortStartIndex = 1;
    if (!(isNotConfig || isNumQpConfigSuccess)) { // 对于dev multiQp cfg非正常场景
        // 合法 portMode 不应在此解析
        if (devCfgPortMode.end()
            != std::find(devCfgPortMode.begin() + afterPortStartIndex, devCfgPortMode.end(), modeValue)) {
            initStatus_ = HcclResult::HCCL_SUCCESS;
        } else {
            HCCL_ERROR(
                "[MulQpInfo][DevCfgMulQpInfoCache][Init]mul qp config invalid, mode[%s] count[%s] ports "
                "[%s]",
                modeValue.c_str(), countValue.c_str(), portValue.c_str());
            initStatus_ = HcclResult::HCCL_E_INTERNAL;
        }
    }
    if (initStatus_ == HcclResult::HCCL_SUCCESS && isNumQpConfigSuccess) {
        cacheInfo_ = qpPorts;
    }
    return initStatus_;
}

bool DevCfgMulQpInfoCache::IsAvailable() const
{
    return initStatus_ == HcclResult::HCCL_SUCCESS && !cacheInfo_.empty();
}

HcclResult DevCfgMulQpInfoCache::GetPortsNumByIpPair(PortNum& portNum, const KeyPair& ipPair) const
{
    (void)ipPair;
    portNum = cacheInfo_.size();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
DevCfgMulQpInfoCache::GetSpecialSourcePortsByIpPair(MulQpSourcePorts& sourcePorts, const KeyPair& ipPair) const
{
    (void)ipPair;
    sourcePorts = cacheInfo_;
    return HcclResult::HCCL_SUCCESS;
}

DevNslbMulQpInfoCache::DevNslbMulQpInfoCache(const NICDeployment nicDeployment, const std::int32_t phyId)
    : nicDeployment_(nicDeployment),
      phyId_(phyId),
      isEnableNslb_(false)
{
    SetMulQpInfoFrom(MUL_QP_FROM::MUL_QP_FROM_DEV_NSLB);
}

HcclResult DevNslbMulQpInfoCache::Init()
{
    std::string modeValue;
    initStatus_ = HrtRaGetHccnCfg(
        static_cast<std::uint32_t>(nicDeployment_), phyId_, HccnCfgKeyT::HCCN_UDP_PORT_MODE, modeValue);
    if (initStatus_ != HcclResult::HCCL_SUCCESS) {
        return initStatus_;
    }

    const bool isNumQpConfigSuccess = !modeValue.empty() && modeValue == "nslb_dp";
    constexpr std::size_t afterPortStartIndex = 2;
    if (initStatus_ == HcclResult::HCCL_SUCCESS && isNumQpConfigSuccess) {
        isEnableNslb_ = true;
    } else if (
        initStatus_ != HcclResult::HCCL_SUCCESS
        || devCfgPortMode.end()
               == std::find(devCfgPortMode.begin() + afterPortStartIndex, devCfgPortMode.end(), modeValue)) {
        initStatus_ = HcclResult::HCCL_E_INTERNAL;
        isEnableNslb_ = false;
        HCCL_ERROR("[MulQpInfo][DevCfgMulQpInfoCache][Init]mul qp config invalid, mode[%s]", modeValue.c_str());
    }
    return initStatus_;
}

bool DevNslbMulQpInfoCache::IsAvailable() const { return initStatus_ == HcclResult::HCCL_SUCCESS && isEnableNslb_; }

HcclResult DevNslbMulQpInfoCache::GetPortsNumByIpPair(PortNum& portNum, const KeyPair& ipPair) const
{
    (void)ipPair;
    portNum = 1;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
DevNslbMulQpInfoCache::GetSpecialSourcePortsByIpPair(MulQpSourcePorts& sourcePorts, const KeyPair& ipPair) const
{
    (void)ipPair;
    sourcePorts = MulQpSourcePorts{static_cast<Port>(hcclNslbDp::GetInstance().Getl4SPortId())};
    return HcclResult::HCCL_SUCCESS;
}

static std::vector<std::string> Split(std::string& s, const std::string& delimiter)
{
    size_t posStart = 0;
    size_t posEnd = s.find(delimiter, posStart);
    std::vector<std::string> res;
    while (posEnd != std::string::npos) {
        std::string token = s.substr(posStart, posEnd - posStart);
        res.push_back(token);
        posStart = posEnd + delimiter.length();
        posEnd = s.find(delimiter, posStart);
    }
    res.push_back(s.substr(posStart));
    return res;
}

static HcclResult GetSrcPortsFromString(
    std::string& s, std::vector<std::uint16_t>& srcPorts, std::uint32_t lineCnt, const std::string& lineAvator)
{
    const std::vector<std::string> strPorts = Split(s, ",");
    srcPorts.resize(strPorts.size(), 0);
    CHK_PRT_RET(
        strPorts.size() > MULTI_QP_CONFIG_SRC_PORT_NUM_MAX || strPorts.empty(),
        HCCL_ERROR(
            "[MulQpInfo][GetSrcPortsFromString][line: %u]config ports num[%u] more than the "
            "threshold[%u].[%s]",
            lineCnt, static_cast<unsigned>(strPorts.size()), MULTI_QP_CONFIG_SRC_PORT_NUM_MAX, lineAvator.c_str()),
        HcclResult::HCCL_E_PARA);

    for (std::uint32_t i = 0; i < strPorts.size(); i++) {
        // 检查端口号是否为全数字的字符串
        CHK_PRT_RET(
            strPorts[i].empty() || DevStrToUint16(strPorts[i], srcPorts[i]) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[MulQpInfo][GetSrcPortsFromString][line: %u]src port[%s]"
                "should be within the range of[1, %u] and configured as a valid integer.[%s]",
                lineCnt, strPorts[i].c_str(), MULTI_QP_CONFIG_SRC_PORT_ID_MAX, lineAvator.c_str()),
            HcclResult::HCCL_E_PARA);
    }
    return HcclResult::HCCL_SUCCESS;
}

EnvConfigPathCache::EnvConfigPathCache() { SetMulQpInfoFrom(MUL_QP_FROM::MUL_QP_FROM_ENV_PORT_CONFIG_PATH); }

HcclResult EnvConfigPathCache::Init()
{
    if (!GetExternalInputQpSrcPortConfigPath().empty()) {
        initStatus_ = LoadMultiQpSrcPortFromFile();
        if (initStatus_ != HcclResult::HCCL_SUCCESS) {
            return initStatus_;
        }
    } else {
        initStatus_ = HcclResult::HCCL_SUCCESS;
    }
    return initStatus_;
}

bool EnvConfigPathCache::IsAvailable() const { return initStatus_ == HcclResult::HCCL_SUCCESS && !cacheInfo_.empty(); }

HcclResult EnvConfigPathCache::LoadMultiQpSrcPortFromFile()
{
    // 读取配置文件
    std::string fileStr = GetExternalInputQpSrcPortConfigPath() + "/MultiQpSrcPort.cfg";
    std::array<char, PATH_MAX> realFile{};
    if (realpath(fileStr.c_str(), realFile.data()) == nullptr) {
        RPT_INPUT_ERR(
            true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>({fileStr, "config file path", "valid absolute path"}));
        HCCL_ERROR(
            "[%s][%s]file[%s] path invalid.", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(),
            fileStr.c_str());
        return HcclResult::HCCL_E_PARA;
    }

    std::ifstream inFile(fileStr.c_str(), std::ifstream::in);
    if (!inFile) {
        RPT_INPUT_ERR(
            true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
            std::vector<std::string>({fileStr, "config file", "file exists and readable"}));
        HCCL_ERROR(
            "[%s][%s]open config file[%s] failed.", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(),
            fileStr.c_str());
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_INFO(
        "[%s][%s]open config file[%s] success.", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(),
        fileStr.c_str());

    // 逐行解析配置文件
    std::uint32_t lineCnt = 1;
    std::string line;
    while (std::getline(inFile, line)) {
        std::string lineAvator = line; // 每行内容的快照, 用于dfx
        // 去除空格和tab
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());

        // 去除注释
        std::string lineInfo = Split(line, "#")[0]; // 只保留#号前的内容
        if (lineInfo.empty()) {
            HCCL_DEBUG(
                "[EnvConfigPathCache][LoadMultiQpSrcPortFromFile][line: %u]comment line, do not parse.[%s]", lineCnt,
                lineAvator.c_str());
            lineCnt++;
            continue;
        }

        // 切分字符串, 检查配置格式
        std::vector<std::string> strIpPort = Split(lineInfo, "=");
        if (strIpPort.size() != MULTI_QP_CONFIG_IP_NUM) {
            const std::string formattedExpect = "[line: " + std::to_string(lineCnt)
                                                + "] Expected format: 'srcIPN,dstIPN=srcPort0,srcPort1,...,srcPortN'";
            RPT_INPUT_ERR(
                true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                std::vector<std::string>({lineInfo, "config line format", formattedExpect}));
            HCCL_ERROR(
                "[%s][%s] %s Config content[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(),
                formattedExpect.c_str(), lineAvator.c_str());
            inFile.close();
            return HcclResult::HCCL_E_PARA;
        }

        // 解析ip对
        std::string ipPair;
        auto ret = GetIpPairFromString(strIpPort[0], ipPair, lineCnt, lineAvator);
        if (ret != HcclResult::HCCL_SUCCESS) {
            RPT_INPUT_ERR(
                true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                std::vector<std::string>({strIpPort[0], "IP pair", "valid IPv4 or IPv6 address"}));
            HCCL_ERROR(
                "[%s][%s] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), "IP format error");
            inFile.close();
            return ret;
        }

        // 解析源端口号
        std::vector<std::uint16_t> srcPorts;
        ret = GetSrcPortsFromString(strIpPort[1], srcPorts, lineCnt, lineAvator);
        if (ret != HcclResult::HCCL_SUCCESS) {
            RPT_INPUT_ERR(
                true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                std::vector<std::string>({strIpPort[1], "Source Ports", "comma-separated list of valid ports"}));
            HCCL_ERROR(
                "[%s][%s] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), "port format error");
            inFile.close();
            return ret;
        }

        // 配置源端口号
        if (cacheInfo_.find(ipPair) != cacheInfo_.end()) {
            const std::string DUPLICATE_IPPAIR_ERROR
                = "[line: " + std::to_string(lineCnt) + "] ip pair: " + ipPair + " has existed";
            RPT_INPUT_ERR(
                true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                std::vector<std::string>({ipPair, "IP pair Key", "unique IP pair without duplicates"}));
            HCCL_ERROR(
                "[%s][%s][line: %u]ip pair[%s] has existed.[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_ENV_CONFIG.c_str(), lineCnt, ipPair.c_str(), lineAvator.c_str());
            inFile.close();
            return HcclResult::HCCL_E_PARA;
        }
        cacheInfo_[ipPair] = srcPorts;

        // 判断文件行数是否超过上限
        if (lineCnt >= MULTI_QP_CONFIG_FILE_LINE_MAX) {
            HCCL_RUN_INFO("[EnvConfigPathCache][LoadMultiQpSrcPortFromFile]config file is too large.");
            break;
        }
        lineCnt++;
    }
    inFile.close();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult EnvConfigPathCache::GetIpPairFromString(
    std::string& s, std::string& ipPair, const std::uint32_t lineCnt, const std::string& lineAvator)
{
    std::vector<std::string> strIps = Split(s, ",");
    CHK_PRT_RET(
        strIps.size() != MULTI_QP_CONFIG_IP_NUM,
        HCCL_ERROR(
            "[EnvConfigPathCache][GetIpPairFromString][line: %u]invalid Ip format.[%s]", lineCnt, lineAvator.c_str()),
        HcclResult::HCCL_E_PARA);

    HcclIpAddress srcIpAddr{};
    // 解析源ip
    auto ret = srcIpAddr.SetReadableAddress(strIps[0]);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[EnvConfigPathCache][GetIpPairFromString][line: %u]srcIp is an invalid format.[%s]", lineCnt,
            lineAvator.c_str()),
        HcclResult::HCCL_E_PARA);

    // 解析目的ip
    HcclIpAddress dstIpAddr{};
    ret = dstIpAddr.SetReadableAddress(strIps[1]);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[EnvConfigPathCache][GetIpPairFromString][line: %u]dstIp is an invalid format.[%s]", lineCnt,
            lineAvator.c_str()),
        HcclResult::HCCL_E_PARA);

    // 记录ip对
    ipPair = s;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult EnvConfigPathCache::GetPortsNumByIpPair(PortNum& portNum, const KeyPair& ipPair) const
{
    const std::string srcIp = std::string(ipPair.first.GetReadableIP());
    const std::string dstIp = std::string(ipPair.second.GetReadableIP());
    // 匹配sip和dip
    std::string pair = srcIp + std::string(",") + dstIp;
    auto iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), portNum = iter->second.size(), HcclResult::HCCL_SUCCESS);
    // 匹配dip
    if (ipPair.first.GetFamily() == AF_INET) {
        pair = std::string("0.0.0.0,") + dstIp;
    } else {
        pair = std::string("::/128,") + dstIp;
    }
    iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), portNum = iter->second.size(), HcclResult::HCCL_SUCCESS);
    // 匹配sip
    if (ipPair.first.GetFamily() == AF_INET) {
        pair = srcIp + std::string(",0.0.0.0");
    } else {
        pair = srcIp + std::string(",::/128");
    }
    iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), portNum = iter->second.size(), HcclResult::HCCL_SUCCESS);
    // 通配
    if (ipPair.first.GetFamily() == AF_INET) {
        pair = std::string("0.0.0.0,0.0.0.0");
    } else {
        pair = std::string("::/128,::/128");
    }
    iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), portNum = iter->second.size(), HcclResult::HCCL_SUCCESS);
    portNum = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult EnvConfigPathCache::GetSpecialSourcePortsByIpPair(MulQpSourcePorts& sourcePorts, const KeyPair& ipPair) const
{
    const std::string srcIp = std::string(ipPair.first.GetReadableIP());
    const std::string dstIp = std::string(ipPair.second.GetReadableIP());
    // 匹配sip和dip
    std::string pair = srcIp + std::string(",") + dstIp;
    auto iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), sourcePorts = iter->second, HcclResult::HCCL_SUCCESS);
    // 匹配dip
    if (ipPair.first.GetFamily() == AF_INET) {
        pair = std::string("0.0.0.0,") + dstIp;
    } else {
        pair = std::string("::/128,") + dstIp;
    }
    iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), sourcePorts = iter->second, HcclResult::HCCL_SUCCESS);
    // 匹配sip
    if (ipPair.first.GetFamily() == AF_INET) {
        pair = srcIp + std::string(",0.0.0.0");
    } else {
        pair = srcIp + std::string(",::/128");
    }
    iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), sourcePorts = iter->second, HcclResult::HCCL_SUCCESS);
    // 通配
    if (ipPair.first.GetFamily() == AF_INET) {
        pair = std::string("0.0.0.0,0.0.0.0");
    } else {
        pair = std::string("::/128,::/128");
    }
    iter = cacheInfo_.find(pair);
    CHK_PRT_RET(iter != cacheInfo_.end(), sourcePorts = iter->second, HcclResult::HCCL_SUCCESS);
    sourcePorts.clear();
    return HcclResult::HCCL_SUCCESS;
}

EnvPerConnectionQpInfoCache::EnvPerConnectionQpInfoCache() : cacheInfo_(0), isSetEnvPerConnectionQp_(false)
{
    SetMulQpInfoFrom(MUL_QP_FROM::MUL_QP_FROM_ENV_PER_CONNECTION);
}

HcclResult EnvPerConnectionQpInfoCache::Init()
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_QPS_PER_CONNECTION, mmSysGetEnvValue);
    std::string rdmaQpsPerConnectionEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    if (rdmaQpsPerConnectionEnv == "EmptyString") {
        return HCCL_SUCCESS;
    }
    cacheInfo_ = GetExternalInputQpsPerConnection();
    isSetEnvPerConnectionQp_ = true;
    initStatus_ = HcclResult::HCCL_SUCCESS;
    return initStatus_;
}

bool EnvPerConnectionQpInfoCache::IsAvailable() const { return isSetEnvPerConnectionQp_ && cacheInfo_ >= 1; }

HcclResult EnvPerConnectionQpInfoCache::GetPortsNumByIpPair(PortNum& portNum, const KeyPair& ipPair) const
{
    (void)ipPair;
    portNum = cacheInfo_;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
EnvPerConnectionQpInfoCache::GetSpecialSourcePortsByIpPair(MulQpSourcePorts& sourcePorts, const KeyPair& ipPair) const
{
    (void)ipPair;
    sourcePorts.resize(cacheInfo_, 0);
    return HcclResult::HCCL_SUCCESS;
}

MulQpInfo::~MulQpInfo()
{
    if (config_) {
        config_.reset();
    }
}

HcclResult MulQpInfo::GetMulQpFromType(MUL_QP_FROM& type)
{
    std::lock_guard<std::mutex> initLock(initLock_);
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(config_);
    type = config_->MulQpInfoFrom();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MulQpInfo::GetPortsNumByIpPair(PortNum& portNum, const KeyPair& ipPair)
{
    std::lock_guard<std::mutex> initLock(initLock_);
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(config_);
    return config_->GetPortsNumByIpPair(portNum, ipPair);
}

HcclResult MulQpInfo::GetSpecialSourcePortsByIpPair(MulQpSourcePorts& sourcePorts, const KeyPair& ipPair)
{
    std::lock_guard<std::mutex> initLock(initLock_);
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(config_);
    return config_->GetSpecialSourcePortsByIpPair(sourcePorts, ipPair);
}
} // namespace hccl
