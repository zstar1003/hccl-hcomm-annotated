/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_AICPU_MGR_H
#define CHANNEL_AICPU_MGR_H

#include "common.h"
#include "channel_param.h"
#include "base_transport_lite_impl.h"
#include "topo_matcher.h"
#include "hcclCommDfxLite.h"
#include <unordered_map>
#include <vector>
#include <memory>

class ChannelAicpuMgr {
public:
    ChannelAicpuMgr(hccl::HcclCommDfxLite& dfx, const hccl::HcclTopoInfo& topoInfo);
    ~ChannelAicpuMgr() = default;

    HcclResult AllocChannelResource(HcclChannelUrmaRes* commParam);
    HcclResult Resume(HcclChannelUrmaRes* commParam);
    HcclResult Clean();

private:
    HcclResult InitUrmaChannel(HcclChannelUrmaRes* commParam);
    HcclResult ProcessUrmaRes(HcclChannelUrmaRes* commParam, bool isInit);
    HcclResult ParsePackData(std::vector<char>& data, ChannelHandle& handle);
    HcclResult ResumePackData(std::vector<char>& data, ChannelHandle& handle);
    HcclResult RegisterChannelCacheCallback(ChannelHandle channel);

    std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>> transportMap_;
    hccl::HcclCommDfxLite& dfx_;
    const hccl::HcclTopoInfo& topoInfo_;
};

#endif // CHANNEL_AICPU_MGR_H
