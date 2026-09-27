/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NS_RECOVERY_H
#define NS_RECOVERY_H

#include "hccl_res.h"
#include "hdc_pub.h"
#include "kfc.h"
#include "comm_config_pub.h"

#include <string>
#include <vector>

namespace hccl {

struct NsRecoveryData {
    std::vector<ChannelHandle> channelHandles_;
    std::vector<ChannelHandle> hostChannelHandleList_;
    uint32_t channelNum_;
    std::string commTag_;
};

class NsRecoveryProcessor {
public:
    void SetKfcControlTransfer(
        std::shared_ptr<HDCommunicate> kfcControlTransferH2D, std::shared_ptr<HDCommunicate> kfcStatusTransferD2H);
    void AddNsRecoveryData(
        const CommEngine& engine, const ChannelHandle* const channelHandles,
        const ChannelHandle* const hostChannelHandleList, uint32_t channelNum, const std::string& commTag);
    HcclResult StopLaunch();
    HcclResult Clean();
    HcclResult Resume(aclrtBinHandle binHandle);

private:
    HcclResult ListenBackGround(Hccl::KfcExecStatus& opInfo);
    HcclResult PollStopStatus();

    std::shared_ptr<HDCommunicate> kfcControlTransferH2D_{nullptr};
    std::shared_ptr<HDCommunicate> kfcStatusTransferD2H_{nullptr};
    std::unordered_map<CommEngine, std::vector<NsRecoveryData>> nsRecoveryDatas_;
};

} // namespace hccl

#endif
