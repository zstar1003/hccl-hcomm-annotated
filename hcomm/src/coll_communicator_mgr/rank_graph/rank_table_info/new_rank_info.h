/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NEW_RANK_INFO_H
#define NEW_RANK_INFO_H

#include <vector>
#include <string>

#include "types.h"
#include "nlohmann/json.hpp"
#include "rank_level_info.h"
#include "control_plane.h"
#include "orion_adapter_hccp.h"
#include "rank_table_source.h"
namespace Hccl {
constexpr unsigned int MAX_VALUE_DEVICEID = 512;
constexpr unsigned int DEFAULT_VALUE_TCPPORT = 16666;
constexpr unsigned int MAX_VALUE_TCPPORT = 65535;
constexpr unsigned int MIN_VALUE_TCPPORT = 1;
constexpr unsigned int MAX_LEVEL_lIST = 8;
const IpAddress DEVICE_PORT_KEY_IPADDRESS = IpAddress(); // 用于socketmanager的端口占用，vnic情况下使用
class NewRankInfo {
public:
    NewRankInfo() {};
    ~NewRankInfo() {};

    u32 rankId{0};
    u32 deviceId{0};
    u32 localId{0};
    u32 replacedLocalId{0};
    u32 devicePort{DEFAULT_VALUE_TCPPORT};
    u32 hostPort{DEFAULT_VALUE_TCPPORT};
    std::vector<RankLevelInfo> rankLevelInfos{};
    ControlPlane controlPlane{};
    TlsStatus tlsStatus{TlsStatus::UNKNOWN};
    TlsStatus hostDpuTlsStatus{TlsStatus::UNKNOWN};
    std::string Describe() const;
    void Deserialize(const nlohmann::json& newRankInfoJson, RankTableSource source = RankTableSource::RANKTABLE);
    explicit NewRankInfo(BinaryStream& binStream);
    void GetBinStream(bool isContainLoaId, BinaryStream& binStream) const;

private:
    void DeserializeRankIdAndLocalId(const nlohmann::json& newRankInfoJson, RankTableSource source);
    void DeserializeDeviceIdAndPort(const nlohmann::json& newRankInfoJson, RankTableSource source);
    void DeserializeLevelList(const nlohmann::json& newRankInfoJson, RankTableSource source);
};

} // namespace Hccl

#endif // NEW_RANK_INFO_H
