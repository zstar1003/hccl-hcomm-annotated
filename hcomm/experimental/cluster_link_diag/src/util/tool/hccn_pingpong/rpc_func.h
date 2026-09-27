/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <thread>
#include <string>
#include <vector>
#include <stdint.h>

extern std::vector<std::thread> rpc_threads;

void hccn_pingpong_init_local_devices_for_rpc(const std::vector<int>& dev_ids);

int pinglist_insert_muti(
    std::string src_devIp, std::vector<std::string> target_devIp_list, std::vector<int> src_hccn_ports,
    std::vector<int> src_hccn_ports_rx, int rx_num);

std::vector<std::vector<uint64_t>> ud_pingpong_tx_muti(std::string src_devIp, int times);

enum PingpongResult { P90Lat = 0, P99Lat = 1, Mean = 2, Pass = 3, Size = 4 };
static const int PingPongInvalidResult = 10000;
