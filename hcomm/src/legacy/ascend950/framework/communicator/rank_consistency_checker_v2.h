/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_CONSISTENCY_CHECKER_V2_H
#define RANK_CONSISTENCY_CHECKER_V2_H

#include <string>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <hccl/hccl_types.h>
#include "hccl/base.h"

namespace hccl {
constexpr u32 CANN_VERSION_MAX_LEN = 50;  // CANN版本最大长度
static constexpr u32 MAX_CRC_LEN_V2 = 16; // A5最大CRC个数

struct CrcEntryV2 {
    std::string name; // 对比的crc名（如"HCCL_BUFFSIZE"）
    u32 crc = 0;
    CrcEntryV2() = default;
    CrcEntryV2(std::string n, u32 c) : name(n), crc(c) {}
};

struct CheckFrameV2 {
    u32 rankTableCrcNum = 0;                     // ranktable CRC个数
    u32 rankTableCrcArray[MAX_CRC_LEN_V2] = {0}; // ranktable CRC数组
    char version[CANN_VERSION_MAX_LEN + 1] = {0};
};

class RankConsistencyCheckerV2 {
public:
    ~RankConsistencyCheckerV2();
    static RankConsistencyCheckerV2& GetInstance(const s32& deviceLogicId);

    HcclResult RecordRankTableCrcV2(u32 crc);
    HcclResult RecordCannVersionV2(const std::string& version);
    HcclResult GenerateCheckFrameV2(CheckFrameV2& localFrame);
    HcclResult CompareCheckFrameV2(const CheckFrameV2& local, const CheckFrameV2& remote);
    u64 GetCheckFrameLengthV2() const;

    void SetInconsistentCheckFirstDone(
        bool inconsistentCheckFirstDone); // 用于标识env_config中InconsistentCheckModel::FIRST时是否已校验
    bool GetInconsistentCheckFirstDone() const;

private:
    HcclResult CalcRawDataCrc(const void* ptr, u64 length, u32& crc) const;
    bool CompareCrcArrayV2(
        const u32* localArray, const u32* remoteArray, const std::vector<CrcEntryV2>& nameSource,
        const std::string& categoryLabel) const;
    HcclResult CompareVersionV2(const CheckFrameV2& local, const CheckFrameV2& remote, bool& isDiff) const;

    // cann 版本号
    char cannVersion_[CANN_VERSION_MAX_LEN + 1] = {0};
    std::vector<CrcEntryV2> rankTableCrcV2_;  // A5 ranktable CRC
    bool inconsistentCheckFirstDone_ = false; // 是否完成首次校验
};
} // namespace hccl
#endif /* RANK_CONSISTENCY_CHECKER_V2_H */
