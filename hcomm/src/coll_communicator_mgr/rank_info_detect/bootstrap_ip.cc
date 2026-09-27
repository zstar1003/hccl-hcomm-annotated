/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bootstrap_ip.h"
#include <mutex>
#include <algorithm>
#include "whitelist.h"
#include "env_config/env_config_v2.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"

namespace Hccl {

static UniversalConcurrentMap<u32, IpAddress> bootstrapIps; // <devPhyId, ip>

std::vector<IpAddress> GetHostSocketWhitelist()
{
    std::vector<IpAddress> hostSocketWhitelist{};

    // 文件路径在g_externalInput.hcclWhiteList已经做过合法性判断, 无需再次校验
    std::string fileName = EnvConfig::GetInstance().GetHostNicConfig().GetWhiteListFile();
    CHK_PRT_THROW(
        fileName.empty(),
        HCCL_ERROR(
            "[%s] HCCL_WHITELIST_DISABLE variable is [0],but "
            "HCCL_WHITELIST_FILE is not set",
            __func__),
        InternalException, "whiteList file name empty");

    Whitelist::GetInstance().LoadConfigFile(fileName);
    Whitelist::GetInstance().GetHostWhiteList(hostSocketWhitelist);
    CHK_PRT_THROW(
        hostSocketWhitelist.empty(),
        HCCL_ERROR("[%s] whitelist file[%s] have no valid host ip.", __func__, fileName.c_str()), InternalException,
        "whiteList invalid");

    HCCL_INFO(
        "[%s] get host socket whitelist success. there are %zu host ip in the whitelist.", __func__,
        hostSocketWhitelist.size());
    return hostSocketWhitelist;
}

void GetAllValidHostIfInfos(
    const std::vector<std::pair<std::string, IpAddress>>& hostIfInfos,
    vector<std::pair<std::string, IpAddress>>& ifInfos)
{
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        auto whitelist = GetHostSocketWhitelist();
        for (auto& ifInfo : hostIfInfos) {
            auto iter = find(whitelist.begin(), whitelist.end(), ifInfo.second);
            if (iter != whitelist.end()) {
                ifInfos.push_back({ifInfo.first, ifInfo.second});
            }
        }
    } else {
        ifInfos = hostIfInfos;
    }
}

bool FindHostIpbyControlIfIp(const std::vector<std::pair<std::string, IpAddress>>& hostIfInfos, IpAddress& ipAddress)
{
    // 匹配指定IP的网卡信息
    auto it = std::find_if(hostIfInfos.begin(), hostIfInfos.end(), [&ipAddress](const auto& hostIfInfo) {
        return hostIfInfo.second == ipAddress;
    });
    if (it != hostIfInfos.end()) {
        HCCL_INFO(
            "[%s] find hostIp success, name[%s] ip[%s]", __func__, it->first.c_str(), ipAddress.GetIpStr().c_str());
        return true;
    }
    RPT_INPUT_ERR(
        true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
        std::vector<std::string>(
            {ipAddress.GetIpStr(), "HCCL_IF_IP", "a valid IP from the available network interfaces"}));
    HCCL_ERROR(
        "[%s][%s] Env config \"HCCL_IF_IP\" is [%s] which is not found in the nic list.",
        LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), ipAddress.GetIpStr().c_str());
    return false;
}

bool FindHostIpByIfName(
    const std::vector<std::pair<std::string, IpAddress>>& hostIfInfos, s32 family, IpAddress& ipAddress)
{
    HCCL_INFO("[%s] find host ip start. family[%d]", __func__, family);

    // 使用Host网卡名和环境变量HCCL_SOCKET_IFNAME配置的网卡名进行比较
    auto socketIfName = EnvConfig::GetInstance().GetHostNicConfig().GetSocketIfName();
    for (auto& hostIfInfo : hostIfInfos) {
        if (hostIfInfo.second.GetFamily() != family) {
            continue;
        }
        u32 matchLen = hostIfInfo.first.size();
        bool configIfNamesFlag = false;
        for (u32 i = 0; i < socketIfName.configIfNames.size(); i++) {
            matchLen = socketIfName.searchExact ? hostIfInfo.first.size() : socketIfName.configIfNames[i].size();
            if (hostIfInfo.first.compare(0, matchLen, socketIfName.configIfNames[i], 0, matchLen) == 0) {
                configIfNamesFlag = true;
            }
        }
        if (configIfNamesFlag != socketIfName.searchNot) {
            configIfNamesFlag = false;
            ipAddress = hostIfInfo.second;
            HCCL_INFO(
                "[%s] find hostIp success by ifName. name[%s] ip[%s]", __func__, hostIfInfo.first.c_str(),
                hostIfInfo.second.GetIpStr().c_str());
            return true;
        }
    }

    HCCL_WARNING("[%s] find host ip fail by family[%d] ifName.", __func__, family);
    return false;
}

bool FindHostIpByIfName(const std::vector<std::pair<std::string, IpAddress>>& hostIfInfos, IpAddress& ipAddress)
{
    s32 socketFamily = EnvConfig::GetInstance().GetSocketConfig().GetSocketFamily();
    socketFamily = (socketFamily == -1) ? AF_INET : socketFamily;
    bool ret = FindHostIpByIfName(hostIfInfos, socketFamily, ipAddress);
    if (!ret) {
        socketFamily = (socketFamily == AF_INET) ? AF_INET6 : AF_INET;
        ret = FindHostIpByIfName(hostIfInfos, socketFamily, ipAddress);
    }
    return ret;
}

bool FindHostIpFromOneNicClass(
    const std::map<std::string, std::map<std::string, IpAddress>>& nicClassifyInfo, const std::string& nicClass,
    IpAddress& ip)
{
    auto iterClass = nicClassifyInfo.find(nicClass);
    if (iterClass != nicClassifyInfo.end()) {
        if (iterClass->second.empty()) {
            HCCL_WARNING("[%s] nic class[%s]: no valid ip.", __func__, nicClass.c_str());
            return false;
        }
        ip = iterClass->second.begin()->second;
        HCCL_INFO(
            "[%s] find host ip success by nic class[%s]. host ifName[%s] ip[%s]", __func__, nicClass.c_str(),
            iterClass->second.begin()->first.c_str(), ip.GetIpStr().c_str());
        return true;
    }
    return false;
}

bool FindHostIPByNicClass(
    const std::vector<std::pair<std::string, IpAddress>>& hostIfInfos, s32 family, IpAddress& ipAddress)
{
    std::map<std::string, std::map<std::string, IpAddress>> nicClassify;
    for (auto& hostIfInfo : hostIfInfos) {
        if (hostIfInfo.second.GetFamily() != family) {
            continue;
        }
        if (hostIfInfo.first.find("lo") == 0) {
            nicClassify["lo"].insert({hostIfInfo.first, hostIfInfo.second});
        } else if (hostIfInfo.first.find("docker") == 0) {
            nicClassify["docker"].insert({hostIfInfo.first, hostIfInfo.second});
        } else {
            nicClassify["normal"].insert({hostIfInfo.first, hostIfInfo.second});
        }
        HCCL_DEBUG(
            "[%s] ifName[%s] addr[%s]", __func__, hostIfInfo.first.c_str(), hostIfInfo.second.GetIpStr().c_str());
    }

    if (FindHostIpFromOneNicClass(nicClassify, "normal", ipAddress)) {
        HCCL_INFO("[%s] find host ip success by nic class[normal]. ip[%s].", __func__, ipAddress.GetIpStr().c_str());
        return true;
    } else if (FindHostIpFromOneNicClass(nicClassify, "docker", ipAddress)) {
        HCCL_INFO("[%s] find host ip success by nic class[docker]. ip[%s].", __func__, ipAddress.GetIpStr().c_str());
        return true;
    } else if (FindHostIpFromOneNicClass(nicClassify, "lo", ipAddress)) {
        HCCL_INFO("[%s] find host ip success by nic class[lo]. ip[%s].", __func__, ipAddress.GetIpStr().c_str());
        return true;
    }

    HCCL_WARNING("[%s] find hostIp by nic class[normal_docker_lo] fail.", __func__);
    return false;
}

bool FindHostIPByNicClass(const std::vector<std::pair<std::string, IpAddress>>& hostIfInfos, IpAddress& ipAddress)
{
    s32 socketFamily = EnvConfig::GetInstance().GetSocketConfig().GetSocketFamily();
    socketFamily = (socketFamily == -1) ? AF_INET : socketFamily;
    bool ret = FindHostIPByNicClass(hostIfInfos, socketFamily, ipAddress);
    if (!ret) {
        socketFamily = (socketFamily == AF_INET) ? AF_INET6 : AF_INET;
        ret = FindHostIPByNicClass(hostIfInfos, socketFamily, ipAddress);
    }
    return ret;
}

bool FindLocalHostIp(const std::vector<std::pair<std::string, IpAddress>>& hostIfInfos, IpAddress& ipAddress)
{
    // 根据HCCL_IF_IP查询ips中匹配的LocalHostIP
    ipAddress = EnvConfig::GetInstance().GetHostNicConfig().GetControlIfIp();
    if (!ipAddress.IsInvalid()) {
        return FindHostIpbyControlIfIp(hostIfInfos, ipAddress);
    }

    // 根据HCCL_SOCKET_IFNAME查询ips中匹配的LocalHostIP
    auto ifnames = EnvConfig::GetInstance().GetHostNicConfig().GetSocketIfName();
    if (!ifnames.configIfNames.empty()) {
        bool ret = FindHostIpByIfName(hostIfInfos, ipAddress);
        if (!ret) {
            RPT_INPUT_ERR(
                true, "EI0001", std::vector<std::string>({"value", "env", "expect"}),
                std::vector<std::string>(
                    {ifnames.configIfNameStr, "HCCL_SOCKET_IFNAME",
                     "a valid network interface name from the available interfaces"}));
            HCCL_ERROR(
                "[%s][%s] Env config \"HCCL_SOCKET_IFNAME\" is [%s] which is not found in the nic list",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_ENV_CONFIG.c_str(), ifnames.configIfNameStr.c_str());
            for (auto& ifInfo : hostIfInfos) {
                HCCL_ERROR(
                    "[%s][%s] get host ip fail by socket Ifname. nic name[%s] ip[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                    LOG_KEYWORDS_ENV_CONFIG.c_str(), ifInfo.first.c_str(), ifInfo.second.Describe().c_str());
            }
        }
        return ret;
    }

    // 不匹配的话以此类推选择normal/docker/lo类型的
    return FindHostIPByNicClass(hostIfInfos, ipAddress);
}

const IpAddress& GetBootstrapIp(u32 devPhyId)
{
    // 如果已获取过则直接使用ip
    auto it = bootstrapIps.Find(devPhyId);
    if (it.second) {
        CHK_PRT_RET(
            !it.first->second.IsInvalid(),
            HCCL_INFO("[%s] hostIp[%s] already exists.", __func__, it.first->second.GetIpStr().c_str()),
            it.first->second);
    }

    // 获取网卡ip
    auto hostIfInfos = HrtGetHostIf(devPhyId);
    CHK_PRT_THROW(
        hostIfInfos.empty(), HCCL_ERROR("[%s] there is no host if.", __func__), InternalException, "get host ip error");

    // 若白名单使能则过滤
    vector<std::pair<std::string, IpAddress>> ifInfos;
    GetAllValidHostIfInfos(hostIfInfos, ifInfos);
    CHK_PRT_THROW(
        ifInfos.empty(), HCCL_ERROR("[%s] there is no valid host if in whitelist.", __func__), InternalException,
        "get host ip error");

    // 获得有效的bootstrapIp
    IpAddress ipAddress{};
    bool ret = FindLocalHostIp(ifInfos, ipAddress);
    CHK_PRT_THROW(!ret, HCCL_ERROR("[%s] there is no valid host ip.", __func__), InternalException, "no valid host ip");

    // 保存有效的bootstrapIp且返回
    bootstrapIps[devPhyId] = ipAddress;

    HCCL_INFO("[%s] get hostIp success. ipAddress[%s]", __func__, ipAddress.GetIpStr().c_str());
    return bootstrapIps[devPhyId];
}

} // namespace Hccl
