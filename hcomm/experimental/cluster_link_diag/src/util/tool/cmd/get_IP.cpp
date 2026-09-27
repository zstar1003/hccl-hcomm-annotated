/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "get_IP.h"

std::vector<std::string> get_ethernet_IPv4_addresses(std::function<bool(std::string)> filter)
{
    std::vector<std::string> ipAddresses;
    struct ifaddrs *ifaddr, *ifa;

    // 获取网络接口列表
    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return ipAddresses; // 返回空向量
    }

    // 遍历所有网络接口
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr)
            continue;

        // 只处理 IPv4 地址
        if (ifa->ifa_addr->sa_family == AF_INET) {
            // 检查是否为以太网接口 (eth* 或 en*)
            std::string ifaceName(ifa->ifa_name);
            if ((filter == nullptr && (ifaceName.find("eth") == 0 || ifaceName.find("en") == 0))
                || (filter != nullptr && filter(ifaceName))) {
                // 获取 IPv4 地址
                struct sockaddr_in* sa = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
                char ipStr[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &(sa->sin_addr), ipStr, INET_ADDRSTRLEN);

                ipAddresses.emplace_back(ipStr);
            }
        }
    }

    // 释放接口列表
    freeifaddrs(ifaddr);
    return ipAddresses;
}
