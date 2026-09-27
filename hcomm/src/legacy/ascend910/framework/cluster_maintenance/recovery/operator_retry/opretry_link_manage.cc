/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opretry_link_manage.h"

namespace hccl {

OpretryLinkManage& OpretryLinkManage::GetInstance(s32 deviceLogicID)
{
    static OpretryLinkManage opretryLinkManage[MAX_DEV_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_DEV_NUM) {
        HCCL_WARNING("[OpretryLinkManage][GetInstance] deviceLogicID[%d] is invalid", deviceLogicID);
        return opretryLinkManage[0];
    }
    return opretryLinkManage[deviceLogicID];
}

OpretryLinkManage::~OpretryLinkManage()
{
    isDeInit_ = true;
    allRemoteRankList_.clear();
    groupAllRemoteRankList_.clear();
}

HcclResult OpretryLinkManage::AddLinkInfoByIdentifier(
    const std::string& identifier, const std::string& newTag, std::vector<u32>& remoteRankList, bool incre)
{
    std::unique_lock<std::mutex> lock(opretryLinkMutex_);
    const auto& identifierIt = allRemoteRankList_.find(identifier);
    if (identifierIt != allRemoteRankList_.end()) {
        const auto& tagIt = identifierIt->second.find(newTag);
        if (tagIt == identifierIt->second.end()) {
            identifierIt->second.emplace(newTag, remoteRankList);
            groupAllRemoteRankList_[identifier].insert(remoteRankList.begin(), remoteRankList.end());
        } else if (incre) {
            // 增量建链场景
            for (auto remoteRank : remoteRankList) {
                if (std::find(tagIt->second.begin(), tagIt->second.end(), remoteRank) == tagIt->second.end()) {
                    tagIt->second.push_back(remoteRank);
                }
            }
            groupAllRemoteRankList_[identifier].insert(remoteRankList.begin(), remoteRankList.end());
        } else {
            // tag已存在，则不重复添加
            HCCL_INFO(
                "[OpretryLinkManage][AddLinkInfoByIdentifier]identifier[%s] newTag[%s] is already add",
                identifier.c_str(), newTag.c_str());
            return HCCL_SUCCESS;
        }
    } else {
        std::unordered_map<std::string, std::vector<u32>> tmp = {{newTag, remoteRankList}};
        allRemoteRankList_.emplace(identifier, tmp);
        std::unordered_set<u32> remoteRankSet(remoteRankList.begin(), remoteRankList.end());
        groupAllRemoteRankList_.emplace(identifier, remoteRankSet);
    }
    return HCCL_SUCCESS;
}

HcclResult OpretryLinkManage::GetLinkInfoByIdentifier(
    const std::string& identifier, const std::string& newTag, std::vector<u32>& remoteRankList)
{
    std::unique_lock<std::mutex> lock(opretryLinkMutex_);
    const auto& identifierIt = allRemoteRankList_.find(identifier);
    if (identifierIt != allRemoteRankList_.end()) {
        const auto& tagIt = identifierIt->second.find(newTag);
        if (tagIt != identifierIt->second.end()) {
            remoteRankList = tagIt->second;
            HCCL_RUN_INFO(
                "[OpretryLinkManage][GetLinkInfoByIdentifier]identifier[%s] newTag[%s] get success", identifier.c_str(),
                newTag.c_str());
            return HCCL_SUCCESS;
        } else {
            HCCL_ERROR("[OpretryLinkManage]newTag[%s] not found, please add it before", newTag.c_str());
            return HCCL_E_PARA;
        }
    } else {
        HCCL_ERROR("[OpretryLinkManage]identifier[%s] not found, please add it before", identifier.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult OpretryLinkManage::GetLinkInfoByIdentifier(const std::string& identifier, std::vector<u32>& remoteRankList)
{
    std::unique_lock<std::mutex> lock(opretryLinkMutex_);
    const auto& identifierIt = groupAllRemoteRankList_.find(identifier);
    if (identifierIt != groupAllRemoteRankList_.end()) {
        remoteRankList.assign(identifierIt->second.begin(), identifierIt->second.end());
        HCCL_RUN_INFO("[OpretryLinkManage][GetLinkInfoByIdentifier]identifier[%s] get success", identifier.c_str());
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[OpretryLinkManage]identifier[%s] not found, please add it before", identifier.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult OpretryLinkManage::DeleteLinkInfoByIdentifier(const std::string& identifier)
{
    CHK_PRT_RET(isDeInit_ == true, HCCL_WARNING("OpretryLinkManage has been destroyed"), HCCL_SUCCESS);
    std::unique_lock<std::mutex> lock(opretryLinkMutex_);
    if (allRemoteRankList_.find(identifier) != allRemoteRankList_.end()) {
        allRemoteRankList_.erase(identifier);
    }
    if (groupAllRemoteRankList_.find(identifier) != groupAllRemoteRankList_.end()) {
        groupAllRemoteRankList_.erase(identifier);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
