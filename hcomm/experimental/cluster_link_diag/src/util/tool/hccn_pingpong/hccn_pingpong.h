/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCN_PING_MY_H
#define HCCN_PING_MY_H

#include <string>
#include <cstdint>
#include <vector>

#include <hccl/hccn_rping.h>

typedef unsigned int u32;

extern u32 _sl;
extern u32 _tc;

int hccn_pingpong_init(
    u32 devId, const std::string& devIp, bool keepAlive = false); // 初始化当前 devId 的唯一 rping ctx
bool hccn_pingpong_is_initialized(u32 devId, bool keepAlive = false);
void hccn_pingpong_cleanup();

int hccn_pingpong_set_targets(
    u32 devId, const std::string& srcDevIp, const std::vector<std::string>& targetDevIpList,
    const std::vector<int>& srcPorts); // 初始化当前 dev ctx 后直接 AddTarget

std::vector<std::vector<HccnRpingResultInfo>> hccn_pingpong_batch_ping(
    u32 devId,
    int times); // 复用 AddTarget 时缓存的 targetInfo，只反复 BatchPing/GetResult

void set_hccn_pingpong_local_log(bool enabled, const std::string& outputDir);
void set_hccn_pingpong_payload_len(u32 payloadLen);
u32 get_hccn_pingpong_payload_len();
void set_hccn_pingpong_interval_ms(u32 intervalMs);
u32 get_hccn_pingpong_interval_ms();

std::vector<std::vector<uint64_t>>
hccn_pingpong_to_rpc_result(const std::vector<std::vector<HccnRpingResultInfo>>&
                                hccnResults); //  把 HCCN 原始结果转成 RPC 需要的 vector<vector<uint64_t>>

#endif
