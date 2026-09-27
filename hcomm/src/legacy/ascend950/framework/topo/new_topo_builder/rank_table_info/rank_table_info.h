/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NEW_RANK_TABLED_H
#define NEW_RANK_TABLED_H

#include <vector>
#include <string>
#include <unordered_map>
#include <climits>
#include "nlohmann/json.hpp"
#include "new_rank_info.h"
#include "port.h"

namespace Hccl {
constexpr unsigned int MAX_RANKCOUNT = 65536;
constexpr u32 UNDEFIEND_LOCAL_ID = UINT_MAX;
class RankTableInfo {
public:
    RankTableInfo() {};
    RankTableInfo(BinaryStream& binaryStream);
    std::string version;
    u32 rankCount{0};
    std::vector<NewRankInfo> ranks;
    bool detour = false;
    std::string Describe() const;
    void Dump() const;
    void Deserialize(const nlohmann::json& rankTableInfoJson, bool isCheck = true);
    vector<char> GetUniqueId(bool isContainLocId) const; // 获取rankTable的字节流，供一致性校验crc时带localId使用
    void GetBinStream(bool isContainLocId, BinaryStream& binaryStream) const;
    void Check();
    void UpdateRankTable(const RankTableInfo& localRankInfo);
    std::unordered_map<u32, std::unordered_map<IpAddress, u32>> GetRankDeviceListenPortMap();

private:
    void CheckAndInsert(
        const std::string& levelId, u32 rankAddrSize, std::unordered_map<std::string, u32>& idRankSizeMap) const;
    void InsertToRank(
        const std::string& levelId, u32 rankAddrSize, std::vector<std::unordered_map<std::string, u32>>& rankLists,
        u32 levelNum) const;
};

} // namespace Hccl

#endif // NEW_RANK_TABLED_H
