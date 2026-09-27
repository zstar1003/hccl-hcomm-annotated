/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_transport_lite_mgr.h"
#include "binary_stream.h"

namespace Hccl {

bool MemTransportLiteMgr::IsOpbaseExist(const LinkData& linkData)
{
    if (opBaseTranspMap.find(linkData) == opBaseTranspMap.end()) {
        return false;
    }
    return true;
}

MemTransportLite* MemTransportLiteMgr::GetOpbase(const LinkData& linkData)
{
    if (UNLIKELY(opBaseTranspMap.find(linkData) == opBaseTranspMap.end())) {
        HCCL_WARNING("OpBase linkData=%s find transport is null", linkData.Describe().c_str());
        return nullptr;
    }
    return opBaseTranspMap[linkData].get();
}

void MemTransportLiteMgr::Reset()
{
    HCCL_INFO("Reset OpbaseTransport");
    opBaseTranspMap.clear();
    for (auto& it : offloadTranspMap) {
        HCCL_INFO("Reset OffloadTransport opTag=%s", it.first.c_str());
        offloadTranspMap[it.first].clear();
    }
    offloadTranspMap.clear();
}

void MemTransportLiteMgr::ParseOpbasePackedData(std::vector<char>& data)
{
    u32 mapSize;
    BinaryStream binaryStream(data);
    binaryStream >> mapSize;

    for (u32 idx = 0; idx < mapSize; idx++) {
        std::vector<char> linkUniqueId;
        binaryStream >> linkUniqueId;
        LinkData link(linkUniqueId);

        std::vector<char> transpUniqueId;
        binaryStream >> transpUniqueId;

        if (!IsOpbaseExist(link)) {
            auto transportCallbackLite = MemTransportCallbackLite(link, *mirrorTaskMgrLite_);
            auto lite = std::make_unique<MemTransportLite>(transpUniqueId, transportCallbackLite);
            HCCL_INFO("Build New OpBase Link=%s, transport=%s", link.Describe().c_str(), lite->Describe().c_str());
            opBaseTranspMap[link] = std::move(lite);
        }
    }
}

MemTransportLite* MemTransportLiteMgr::GetOffload(const std::string& opTag, const LinkData& linkData)
{
    if (UNLIKELY(
            offloadTranspMap.find(opTag) == offloadTranspMap.end()
            || offloadTranspMap[opTag].find(linkData) == offloadTranspMap[opTag].end())) {
        HCCL_WARNING(
            "offload opTag=%s, linkData=%s find transport is null", opTag.c_str(), linkData.Describe().c_str());
        return nullptr;
    }
    return offloadTranspMap[opTag][linkData].get();
}

void MemTransportLiteMgr::ParseOffloadPackedData(const std::string& opTag, std::vector<char>& data)
{
    u32 mapSize;
    BinaryStream binaryStream(data);
    binaryStream >> mapSize;

    HCCL_INFO("Build New Offload OpTag=%s transports", opTag.c_str());
    for (u32 idx = 0; idx < mapSize; idx++) {
        std::vector<char> linkUniqueId;
        binaryStream >> linkUniqueId;
        LinkData link(linkUniqueId);

        std::vector<char> transpUniqueId;
        binaryStream >> transpUniqueId;
        auto transportCallbackLite = MemTransportCallbackLite(link, *mirrorTaskMgrLite_);
        auto lite = std::make_unique<MemTransportLite>(transpUniqueId, transportCallbackLite);
        HCCL_INFO(
            "MemTransportLiteMgr::ParseOffloadPackedData: %s, %s", link.Describe().c_str(), lite->Describe().c_str());
        offloadTranspMap[opTag][link] = std::move(lite);
    }
}

void MemTransportLiteMgr::ParseOpbaseAllPackedData(BinaryStream& binaryStream)
{
    u32 opbasedMapSize;
    binaryStream >> opbasedMapSize;

    for (u32 idx = 0; idx < opbasedMapSize; idx++) {
        std::vector<char> linkUniqueId;
        binaryStream >> linkUniqueId;
        LinkData link(linkUniqueId);

        std::vector<char> transpUniqueId;
        binaryStream >> transpUniqueId;

        if (!IsOpbaseExist(link)) {
            auto transportCallbackLite = MemTransportCallbackLite(link, *mirrorTaskMgrLite_);
            auto lite = std::make_unique<MemTransportLite>(transpUniqueId, transportCallbackLite);
            HCCL_INFO("Build New OpBase Link=%s, transport=%s", link.Describe().c_str(), lite->Describe().c_str());
            opBaseTranspMap[link] = std::move(lite);
        }
    }
}

void MemTransportLiteMgr::ParseOffloadAllPackedData(BinaryStream& binaryStream)
{
    u32 opTagNum;
    binaryStream >> opTagNum;
    HCCL_INFO("ParseOffloadAllPackedData: opTagNum=%u", opTagNum);
    for (u32 idx = 0; idx < opTagNum; idx++) {
        std::vector<char> opTagVec;
        binaryStream >> opTagVec;
        std::string opTag(opTagVec.begin(), opTagVec.end());
        HCCL_INFO("ParseOffloadAllPackedData: opTag=%s", opTag.c_str());
        u32 offloadMapSize;
        binaryStream >> offloadMapSize;
        for (u32 j = 0; j < offloadMapSize; j++) {
            std::vector<char> linkUniqueId;
            binaryStream >> linkUniqueId;
            LinkData link(linkUniqueId);

            std::vector<char> transpUniqueId;
            binaryStream >> transpUniqueId;
            auto transportCallbackLite = MemTransportCallbackLite(link, *mirrorTaskMgrLite_);
            auto lite = std::make_unique<MemTransportLite>(transpUniqueId, transportCallbackLite);
            HCCL_INFO(
                "MemTransportLiteMgr::ParseOffloadAllPackedData: %s, %s", link.Describe().c_str(),
                lite->Describe().c_str());
            offloadTranspMap[opTag][link] = std::move(lite);
        }
    }
}

void MemTransportLiteMgr::ParseAllPackedData(std::vector<char>& data)
{
    BinaryStream binaryStream(data);
    ParseOpbaseAllPackedData(binaryStream);
    ParseOffloadAllPackedData(binaryStream);
}

} // namespace Hccl
