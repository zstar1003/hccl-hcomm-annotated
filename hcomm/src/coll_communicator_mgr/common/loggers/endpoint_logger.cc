/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_logger.h"
#include "comm_addr_logger.h"
#include "hccl_comm_pub.h"

namespace hcomm {
namespace logger {

    void EndpointLogger::PrintDeviceLocation(uint32_t idx, const char* endpointName, const EndpointLoc& loc)
    {
        HCCL_INFO(
            "[%s] channelDescs[%u] %s loc: locType[%d], devPhyId[%u], superDevId[%u], serverIdx[%u], superPodIdx[%u]",
            __func__, idx, endpointName, loc.locType, loc.device.devPhyId, loc.device.superDevId, loc.device.serverIdx,
            loc.device.superPodIdx);
    }

    void EndpointLogger::PrintHostLocation(uint32_t idx, const char* endpointName, const EndpointLoc& loc)
    {
        HCCL_INFO(
            "[%s] channelDescs[%u] %s loc: locType[%d], host.id[%u]", __func__, idx, endpointName, loc.locType,
            loc.host.id);
    }

    void EndpointLogger::PrintLocation(uint32_t idx, const char* endpointName, const EndpointLoc& loc)
    {
        if (loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            PrintDeviceLocation(idx, endpointName, loc);
        } else if (loc.locType == ENDPOINT_LOC_TYPE_HOST) {
            PrintHostLocation(idx, endpointName, loc);
        } else {
            HCCL_INFO("[%s] channelDescs[%u] %s loc: locType[%d]", __func__, idx, endpointName, loc.locType);
        }
    }

    void EndpointLogger::Print(uint32_t idx, const char* endpointName, const EndpointDesc& endpointDesc)
    {
        // 打印通信地址
        CommAddrLogger::Print(idx, endpointName, endpointDesc.commAddr);

        // 打印位置信息
        PrintLocation(idx, endpointName, endpointDesc.loc);
    }

} // namespace logger
} // namespace hcomm
