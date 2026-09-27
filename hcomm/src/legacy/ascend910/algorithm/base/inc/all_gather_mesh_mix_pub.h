/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MESH_MIX_PUB_H
#define ALL_GATHER_MESH_MIX_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class AllgatherMeshMix : public AlgTemplateBase {
public:
    explicit AllgatherMeshMix(const HcclDispatcher dispatcher);

    ~AllgatherMeshMix() override;

    // should be called soon after template AllGatherMeshMix instance created
    HcclResult Prepare(
        std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 userRank = INVALID_VALUE_RANKID,
        HcomCollOpInfo* opInfo = nullptr, u32 interRank = INVALID_VALUE_RANKID, u32 interRankSize = 0) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
    // 获取向该rank往前的第i个rank
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }

    inline u32 ForwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + step) % rankSize;
    }
    std::vector<Stream> meshStreams_; /** 多steam**/

    std::vector<std::shared_ptr<LocalNotify>>* meshSignal_{nullptr};    /* 每个ring创建一个signal */
    std::vector<std::shared_ptr<LocalNotify>>* meshSignalAux_{nullptr}; /* 从stream wait，主steam record */
    u32 interRank_;                                                     // 在所有rank环上的rankid
    u32 interRankSize_;
    HcomCollOpInfo* opInfo_{nullptr};

private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
};
} // namespace hccl

#endif /* ALL_GATHER_MESH_PUB_H */
