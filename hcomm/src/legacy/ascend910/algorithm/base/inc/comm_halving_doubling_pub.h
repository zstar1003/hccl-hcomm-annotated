/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_HALVING_DOUBLING_PUB_H
#define COMM_HALVING_DOUBLING_PUB_H

#include "comm_base_pub.h"

namespace hccl {
class CommHalvingDoubling : public CommBase {
public:
    explicit CommHalvingDoubling(
        const std::string& collectiveId, const u32 userRank, const u32 userRankSize, const u32 rank, const u32 rankSize,
        const TopoType topoFlag, const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool>& notifyPool,
        std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, const IntraExchanger& exchanger,
        const std::vector<RankInfo> paraVector, const DeviceMem& inputMem, const DeviceMem& outputMem,
        const bool isUsedRdmaLevel0, const std::string& tag = "",
        const NICDeployment nicDeployInner = NICDeployment::NIC_DEPLOYMENT_DEVICE,
        const u32 subUserRankRoot = INVALID_VALUE_RANKID,
        HalvingDoublingType halvingDoublingType = HalvingDoublingType::RECURSIVE_HALVING_DOUBLING,
        const bool isHaveCpuRank = false, const bool useSuperPodMode = false);

    ~CommHalvingDoubling() override;

protected:
    // 计算当前rank与其他rank之间的link个数:server/client两种角色,H-D需要派生类实现
    HcclResult CalcLink() override;
    bool NeedDataReceivedAck() override;

private:
    const u32 subUserRankRoot_;
    HalvingDoublingType halvingDoublingType_;
};
} // namespace hccl

#endif /* COMM_HALVING_DOUBLING_PUB_H */
