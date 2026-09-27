/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_consistency_checker_v2.h"
#include "adapter_pub.h"
#include "adapter_error_manager_pub.h"
#include "checkcrc.h"
using namespace Hccl;

namespace hccl {
RankConsistencyCheckerV2::~RankConsistencyCheckerV2() = default;

RankConsistencyCheckerV2& RankConsistencyCheckerV2::GetInstance(const s32& deviceLogicId)
{
    static RankConsistencyCheckerV2 instance[MAX_MODULE_DEVICE_NUM];
    HCCL_INFO("[RankConsistencyCheckerV2][GetInstance] get deviceLogicId[%d]", deviceLogicId);
    CHK_PRT_RET(
        (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM || deviceLogicId < 0),
        HCCL_WARNING("[RankConsistencyCheckerV2::RankConsistencyCheckerV2]deviceLogicId[%d] is invalid", deviceLogicId),
        instance[0]);
    return instance[deviceLogicId];
}

HcclResult RankConsistencyCheckerV2::RecordRankTableCrcV2(u32 crc)
{
    rankTableCrcV2_.clear();
    rankTableCrcV2_.emplace_back(std::string("ranktable_content"), crc);
    HCCL_INFO("[RankConsistencyCheckerV2::RecordRankTableCrcV2] ranktable crc[%u] recorded.", crc);
    return HCCL_SUCCESS;
}

HcclResult RankConsistencyCheckerV2::RecordCannVersionV2(const std::string& version)
{
    u32 strLen = version.length();
    s32 sRet = memset_s(cannVersion_, CANN_VERSION_MAX_LEN + 1, 0, CANN_VERSION_MAX_LEN + 1);
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_WARNING(
            "[RankConsistencyCheckerV2::RecordCannVersionV2]memory set 0 fail for version str "
            "array. return[%d].",
            sRet),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        strLen == 0, HCCL_WARNING("[RankConsistencyCheckerV2::RecordCannVersionV2] version information str is empty."),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        strLen >= CANN_VERSION_MAX_LEN + 1,
        HCCL_WARNING("[RankConsistencyCheckerV2::RecordCannVersionV2]"
                     "length of version information str is too long."),
        HCCL_SUCCESS);
    sRet = strncpy_s(cannVersion_, CANN_VERSION_MAX_LEN + 1, version.c_str(), strLen);
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_WARNING("[RankConsistencyCheckerV2::RecordCannVersionV2] call strncpy_s failed, return [%d].", sRet),
        HCCL_SUCCESS);
    HCCL_INFO("[RankConsistencyCheckerV2::RecordCannVersionV2] version[%s] recorded.", version.c_str());
    return HCCL_SUCCESS;
}

HcclResult RankConsistencyCheckerV2::GenerateCheckFrameV2(CheckFrameV2& localFrame)
{
    s32 sRet = memset_s(&localFrame, sizeof(CheckFrameV2), 0, sizeof(CheckFrameV2));
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("[RankConsistencyCheckerV2::GenerateCheckFrameV2] memset failed."), HCCL_E_INTERNAL);

    // 填充ranktable CRC
    localFrame.rankTableCrcNum = std::min(static_cast<u32>(rankTableCrcV2_.size()), MAX_CRC_LEN_V2);
    for (u32 i = 0; i < localFrame.rankTableCrcNum; i++) {
        localFrame.rankTableCrcArray[i] = rankTableCrcV2_[i].crc;
    }

    // 填充CANN版本
    sRet = memcpy_s(localFrame.version, CANN_VERSION_MAX_LEN + 1, cannVersion_, CANN_VERSION_MAX_LEN + 1);
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("[RankConsistencyCheckerV2::GenerateCheckFrameV2] memcpy version failed."),
        HCCL_E_INTERNAL);

    HCCL_INFO("[RankConsistencyCheckerV2::GenerateCheckFrameV2] success");
    return HCCL_SUCCESS;
}

HcclResult RankConsistencyCheckerV2::CompareCheckFrameV2(const CheckFrameV2& local, const CheckFrameV2& remote)
{
    // 默认值是false表示相同，只有不同才会赋值true代表校验参数不一致，后续比对方法不会再赋值false
    bool isDiff = false;
    // dfx日志处理在下面函数的内部，外部不处理日志
    // 比对ranktable CRC
    isDiff
        = isDiff || CompareCrcArrayV2(local.rankTableCrcArray, remote.rankTableCrcArray, rankTableCrcV2_, "ranktable");

    CompareVersionV2(local, remote, isDiff);
    return isDiff ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

u64 RankConsistencyCheckerV2::GetCheckFrameLengthV2() const { return sizeof(CheckFrameV2); }

void RankConsistencyCheckerV2::SetInconsistentCheckFirstDone(bool inconsistentCheckFirstDone)
{
    inconsistentCheckFirstDone_ = inconsistentCheckFirstDone;
}

bool RankConsistencyCheckerV2::GetInconsistentCheckFirstDone() const { return inconsistentCheckFirstDone_; }

// private
bool RankConsistencyCheckerV2::CompareCrcArrayV2(
    const u32* localArray, const u32* remoteArray, const std::vector<CrcEntryV2>& nameSource,
    const std::string& categoryLabel) const
{
    bool isDiff = false;
    u32 count = std::min(static_cast<u32>(nameSource.size()), MAX_CRC_LEN_V2);
    for (u32 i = 0; i < count; i++) {
        if (localArray[i] != remoteArray[i]) {
            const std::string& name = nameSource[i].name;
            HCCL_ERROR(
                "[%s][%s] CMD information %s check fail, %s local[%u], remote[%u].", LOG_KEYWORDS_INIT_CHANNEL.c_str(),
                LOG_KEYWORDS_PARAMETER_CONFLICT.c_str(), categoryLabel.c_str(), name.c_str(), localArray[i],
                remoteArray[i]);
            RPT_INPUT_ERR(
                true, "EI0005", std::vector<std::string>({"ccl_op", "para_name", "local_para", "remote_para"}),
                std::vector<std::string>({"N/A", name, std::to_string(localArray[i]), std::to_string(remoteArray[i])}));
            isDiff = true;
        }
    }

    return isDiff;
}

std::vector<std::string_view> Split(const std::string& str, char delimiter)
{
    std::vector<std::string_view> parts;
    std::string_view sv(str);
    size_t start = 0;
    for (size_t i = 0; i <= sv.size(); ++i) {
        if (i == sv.size() || sv[i] == delimiter) {
            parts.push_back(sv.substr(start, i - start));
            start = i + 1;
        }
    }
    return parts;
}

std::vector<std::string> GetVersionErrMessage(const std::string& localVersion, const std::string& remoteVersion)
{
    std::vector<std::string> versionErrMessage;
    auto localVersionParts = Split(localVersion, '_');
    auto remoteVersionParts = Split(remoteVersion, '_');
    std::string localHcommVersion = localVersionParts.size() > 0 ? std::string(localVersionParts[0]) : "empty";
    std::string localHcclVersion = localVersionParts.size() > 1 ? std::string(localVersionParts[1]) : "empty";
    std::string remoteHcommVersion = remoteVersionParts.size() > 0 ? std::string(remoteVersionParts[0]) : "empty";
    std::string remoteHcclVersion = remoteVersionParts.size() > 1 ? std::string(remoteVersionParts[1]) : "empty";

    bool hcommDiff = (localHcommVersion != remoteHcommVersion);
    bool hcclDiff = (localHcclVersion != remoteHcclVersion);
    if (hcommDiff && !hcclDiff) {
        // 将版本号拆分为hcomm版本号和hccl版本号, 构造ErrMessage
        // The %s versions are inconsistent. The local %s, while the remote %s.
        versionErrMessage
            = {"Toolkit (cann-hcomm)", "version is " + localHcommVersion, "version is " + remoteHcommVersion};
    } else if (!hcommDiff && hcclDiff) {
        versionErrMessage = {"ops (cann-hccl)", "version is " + localHcclVersion, "version is " + remoteHcclVersion};
    } else if (hcommDiff && hcclDiff) {
        versionErrMessage
            = {"Toolkit (cann-hcomm) and ops (cann-hccl)",
               "Toolkit (cann-hcomm) version is " + localHcommVersion + " and ops (cann-hccl) version is "
                   + localHcclVersion,
               "Toolkit (cann-hcomm) version is " + remoteHcommVersion + " and ops (cann-hccl) version is "
                   + remoteHcclVersion};
    } else {
        // 拆分版本号字符串失败, 构造为原版的ErrMessage
        // The HCCL versions are inconsistent. The local version is %s, while the remote version is %s.
        HCCL_ERROR("[RankConsistentcyChecker][GetVersionErrMessage]Failed to split the version string.");
        versionErrMessage = {"HCCL", "version is " + localVersion, "version is " + remoteVersion};
    }
    return versionErrMessage;
}

HcclResult
RankConsistencyCheckerV2::CompareVersionV2(const CheckFrameV2& local, const CheckFrameV2& remote, bool& isDiff) const
{
    std::string localVer(local.version);
    std::string remoteVer(remote.version);
    if (localVer.empty() || remoteVer.empty()) {
        HCCL_WARNING(
            "[RankConsistencyCheckerV2::CompareVersionV2] CANN version str is empty. local[%s], remote[%s].",
            localVer.c_str(), remoteVer.c_str());
        return HCCL_SUCCESS;
    } else if (localVer != remoteVer) {
        auto versionErrMessage = GetVersionErrMessage(localVer, remoteVer);
        // GetVersionErrMessage函数内部已保证versionErrMessage.size() == 3
        RPT_INPUT_ERR(
            true, "EI0008", std::vector<std::string>({"inconsistent_package", "local_version", "remote_version"}),
            std::vector<std::string>({versionErrMessage[0], versionErrMessage[1], versionErrMessage[2]}));
        HCCL_ERROR(
            "[%s][%s] CANN version mismatch: local[%s], remote[%s].", LOG_KEYWORDS_INIT_CHANNEL.c_str(),
            LOG_KEYWORDS_VERSION_CONFLICT.c_str(), localVer.c_str(), remoteVer.c_str());
        isDiff = true;
    }
    return HCCL_SUCCESS;
}

HcclResult RankConsistencyCheckerV2::CalcRawDataCrc(const void* ptr, u64 length, u32& crc) const
{
    // 计算内存数据块CRC
    Hccl::CheckCrc checkCrc;
    HcclResult ret = checkCrc.Calc32Crc(static_cast<const char*>(ptr), length, &crc);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[RankConsistencyCheckerV2::CalcRawDataCrc] errNo[0x%016llx] calc string crc error",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[RankConsistencyCheckerV2::CalcRawDataCrc] result crc[%u].", crc);
    return HCCL_SUCCESS;
}
} // namespace hccl
