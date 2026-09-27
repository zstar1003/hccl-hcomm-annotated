/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_CHANNEL_HELPER_H
#define AIV_CHANNEL_HELPER_H

#include <vector>
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"

/**
 * @brief AIV 专用辅助类，提供 AIV 批量建链状态机接口。
 */
class AivChannelHelper {
public:
    static HcclResult HandleStatus(
        const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
        const std::vector<int32_t>& linkStatusList, int32_t* statusList);
    static HcclResult PreAllocChannels(
        ChannelHandle* targetChannels, ChannelHandle* userChannels, const HcommChannelDesc* channelDescs,
        uint32_t channelNum);

private:
    static HcclResult FillDevEntities(
        const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
        const int32_t* linkStatusList);
};

#endif // AIV_CHANNEL_HELPER_H
