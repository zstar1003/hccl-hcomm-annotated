/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_FACTORY_H
#define CHANNEL_FACTORY_H
#include <memory>
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "interface_channel.h"

namespace Hccl {

// 工厂函数声明 输出作为参数传出
HcclResult CreateChannel(
    CommEngine engine, HcclChannelDesc channelDesc, void* addr, uint64_t size, uint32_t localRankId,
    std::unique_ptr<IChannel>& channel)
{
    return HCCL_SUCCESS;
};

// 具体的创建函数声明
std::unique_ptr<IChannel>
CreateDpuRoceChannel(HcclChannelDesc channelDesc, void* addr, uint64_t size, uint32_t localRankId);

} // namespace Hccl
#endif
