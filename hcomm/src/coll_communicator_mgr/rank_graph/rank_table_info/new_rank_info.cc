/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "new_rank_info.h"

#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "json_parser.h"
#include "const_val.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"

namespace Hccl {
void CheakDeviceIdAndDevicePort(u32 deviceId, u32& devicePort)
{
    if (deviceId > MAX_VALUE_DEVICEID) {
        THROW<InvalidParamsException>(
            StringFormat("device_id [%u] is out of range [%u] to [%u]", deviceId, MIN_VALUE_U32, MAX_VALUE_DEVICEID));
    }
    if (devicePort > MAX_VALUE_TCPPORT || devicePort < MIN_VALUE_TCPPORT) {
        THROW<InvalidParamsException>(StringFormat(
            "device_port [%u] is out of range [%u] to [%u]", devicePort, MIN_VALUE_TCPPORT, MAX_VALUE_TCPPORT));
    }
}

void CheakLevelJsonsSize(u64 levelJsonsSize)
{
    if (levelJsonsSize > MAX_LEVEL_lIST) {
        THROW<InvalidParamsException>(
            StringFormat("level_list size [%u], exceeds the maximum limit [%u]", levelJsonsSize, MAX_LEVEL_lIST));
    }
}

void NewRankInfo::DeserializeRankIdAndLocalId(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    std::string msgRankid = "error occurs when parser object of propName \"rank_id\"";
    std::string msgLocalid = "error occurs when parser object of propName \"local_id\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgRankid, (rankId = GetJsonPropertyUInt(newRankInfoJson, "rank_id")), newRankInfoJson,
        "rank_id", "0 ~ UINT32_MAX", source);
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgLocalid, (localId = GetJsonPropertyUInt(newRankInfoJson, "local_id")),
        newRankInfoJson, "local_id", "0 ~ UINT32_MAX", source);
    if (localId > BACKUP_LOCAL_ID) {
        THROW<InvalidParamsException>(
            StringFormat("local_id [%u] is out of range [%u] to [%u]", localId, MIN_VALUE_U32, BACKUP_LOCAL_ID));
    }
    if (localId == BACKUP_LOCAL_ID) {
        std::string msgReplacedId = "error occurs when parser object of propName \"replaced_local_id\"";
        TRY_CATCH_THROW_REPORT(
            InvalidParamsException, msgReplacedId,
            (replacedLocalId = GetJsonPropertyUInt(newRankInfoJson, "replaced_local_id")), newRankInfoJson,
            "replaced_local_id", "0 ~ UINT32_MAX", source);
        if (replacedLocalId > BACKUP_LOCAL_ID - 1) {
            THROW<InvalidParamsException>(StringFormat(
                "replaced_local_id [%u] is out of range [%u] to [%u]", replacedLocalId, MIN_VALUE_U32,
                BACKUP_LOCAL_ID - 1));
        }
    } else {
        replacedLocalId = localId;
    }
}

void NewRankInfo::DeserializeDeviceIdAndPort(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    std::string msgDeviceid = "error occurs when parser object of propName \"device_id\"";
    std::string msgdeviceport = "error occurs when parser object of propName \"device_port\"";
    std::string msghostport = "error occurs when parser object of propName \"host_port\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgDeviceid, (deviceId = GetJsonPropertyUInt(newRankInfoJson, "device_id")),
        newRankInfoJson, "device_id", "0 ~ UINT32_MAX", source);
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgdeviceport,
        (devicePort = GetJsonPropertyUInt(newRankInfoJson, "device_port", false, DEFAULT_VALUE_TCPPORT)),
        newRankInfoJson, "device_port", "0 ~ UINT32_MAX", source);
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msghostport,
        (hostPort = GetJsonPropertyUInt(newRankInfoJson, "host_port", false, DEFAULT_VALUE_TCPPORT)), newRankInfoJson,
        "host_port", "0 ~ UINT32_MAX", source);
    CheakDeviceIdAndDevicePort(deviceId, devicePort);
    if (hostPort > MAX_VALUE_TCPPORT || hostPort < MIN_VALUE_TCPPORT) {
        THROW<InvalidParamsException>(StringFormat(
            "host_port [%u] is out of range [%u] to [%u]", hostPort, MIN_VALUE_TCPPORT, MAX_VALUE_TCPPORT));
    }
}

void NewRankInfo::DeserializeLevelList(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    nlohmann::json levelJsons;
    std::string msgLevellist = "error occurs when parser object of propName \"level_list\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgLevellist, (GetJsonPropertyList(newRankInfoJson, "level_list", levelJsons)),
        newRankInfoJson, "level_list", "array", source);
    CheakLevelJsonsSize(levelJsons.size());
    for (auto& levelJson : levelJsons) {
        RankLevelInfo levelInfo;
        levelInfo.Deserialize(levelJson, source);
        for (auto& addrsInfo : levelInfo.rankAddrs) {
            addrsInfo.socketPort_ = devicePort;
        }
        rankLevelInfos.emplace_back(levelInfo);
    }

    std::vector<u32> levelSequence;
    for (auto& levelInfos : rankLevelInfos) {
        levelSequence.emplace_back(levelInfos.netLayer);
    }

    for (u32 i = 1; i < levelSequence.size(); i++) {
        if (levelSequence[i] <= levelSequence[i - 1]) {
            THROW<InvalidParamsException>(StringFormat(
                "[NewRankInfo::%s] failed with level is not increased "
                "in sequence. rankId[%d], localId[%d], levelSequence[%u]",
                __func__, rankId, localId, levelSequence.size()));
        }
    }

    if (newRankInfoJson.contains("control_plane")) {
        nlohmann::json controlJsons;
        std::string msgControlPlane = "error occurs when parser object of propName \"control_plane\"";
        controlJsons = newRankInfoJson.at("control_plane");
        controlPlane.Deserialize(controlJsons, source);
    }
}

void NewRankInfo::Deserialize(const nlohmann::json& newRankInfoJson, RankTableSource source)
{
    DeserializeRankIdAndLocalId(newRankInfoJson, source);
    DeserializeDeviceIdAndPort(newRankInfoJson, source);
    DeserializeLevelList(newRankInfoJson, source);
}

std::string NewRankInfo::Describe() const
{
    return StringFormat(
        "NewRankInfo[rankId=%d, localId=%d, replacedLocalId=%d, ranklevelInfos size=%d, device_port=%d, "
        "host_port=%d, tlsStatus=%d, hostDpuTlsStatus=%d]",
        rankId, localId, replacedLocalId, rankLevelInfos.size(), devicePort, hostPort, static_cast<int>(tlsStatus),
        static_cast<int>(hostDpuTlsStatus));
}

NewRankInfo::NewRankInfo(BinaryStream& binStream)
{
    binStream >> rankId >> localId >> replacedLocalId >> deviceId >> devicePort >> hostPort;
    HCCL_DEBUG("[NewRankInfo] localId[%d]", localId);
    size_t rankLevelNum;
    binStream >> rankLevelNum;
    for (u32 i = 0; i < rankLevelNum; i++) {
        RankLevelInfo levelInfo(binStream);
        rankLevelInfos.emplace_back(levelInfo);
    }
    ControlPlane controlPlanes(binStream);
    controlPlane = controlPlanes;
    binStream >> tlsStatus;
    binStream >> hostDpuTlsStatus;
}

void NewRankInfo::GetBinStream(bool isContainLoaId, BinaryStream& binStream) const
{
    if (rankLevelInfos.size() == 0) {
        std::string msg = StringFormat("rankLevelInfos size is zero.");
        THROW<InvalidParamsException>(msg);
    }
    if (isContainLoaId) {
        binStream << rankId << localId << replacedLocalId << deviceId << devicePort << hostPort;
    } else {
        binStream << rankId << INVALID_RANKID << INVALID_RANKID << deviceId << devicePort << hostPort;
    }
    binStream << rankLevelInfos.size();
    HCCL_INFO("[NewRankInfo] rankLevelInfos size[%u], rankId[%d]", rankLevelInfos.size(), rankId);
    for (auto& it : rankLevelInfos) {
        it.GetBinStream(binStream);
    }
    controlPlane.GetBinStream(binStream);
    binStream << tlsStatus;
    binStream << hostDpuTlsStatus;
}

} // namespace Hccl
