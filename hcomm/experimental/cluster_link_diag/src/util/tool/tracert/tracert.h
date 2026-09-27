/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRACERT_H
#define TRACERT_H

#include <vector>
#include <string>
#include <utility>
#include <sys/time.h>
#include <netinet/in.h>

inline constexpr int DEFAULT_SRC_PORT = 49152;
inline constexpr int DEFAULT_DST_PORT = 4791;
inline constexpr int MAX_HOPS = 5;

class Tracert {
public:
    explicit Tracert(int src_device_id = 0);
    ~Tracert();

    std::vector<std::vector<std::string>> trace(const std::vector<std::pair<std::string, int>>& target_port_list);

private:
    int src_device_id_;
};

std::vector<std::vector<std::vector<std::string>>>
tracert_ports_multi_by_src_ip(std::string srcDevIp, std::vector<std::string> targets, std::vector<int> port_num);
std::vector<std::vector<std::vector<std::string>>> tracert_ports_multi_by_src_ip_with_sport_begin(
    std::string srcDevIp, std::vector<std::string> targets, std::vector<int> port_num, int sport_begin);
std::vector<std::string> hccn_device_ip_list(int device_count);
std::string hccn_lldp_mgmt_ip(std::string device_id);

#endif // TRACERT_H
