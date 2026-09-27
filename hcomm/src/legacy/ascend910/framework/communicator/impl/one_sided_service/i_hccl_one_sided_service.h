/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef I_HCCL_ONE_SIDED_SERVICE_H
#define I_HCCL_ONE_SIDED_SERVICE_H

#include <memory>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <unordered_set>
#include "hccl_network_pub.h"
#include "hccl_socket.h"
#include "topoinfo_struct.h"
#include "hccl_socket_manager.h"
#include "notify_pool.h"

namespace hccl {
class IHcclOneSidedService {
public:
    IHcclOneSidedService(std::unique_ptr<HcclSocketManager>& socketManager, std::unique_ptr<NotifyPool>& notifyPool);

    virtual ~IHcclOneSidedService() = default;

    // 为了尽可能保障框架依赖兼容性，除了引用以外，参数不通过构造函数传递
    virtual HcclResult Config(
        const HcclDispatcher& dispatcher, const HcclRankLinkInfo& localRankInfo, const RankTable_t* rankTable,
        std::string identifier = "", bool isStandardCard = false, const std::unordered_set<u32>& enableP2PRankIds = {});

    // 析构时要处理的逻辑
    virtual HcclResult DeInit();

    virtual HcclResult SetNetDevCtx(const HcclNetDevCtx& netDevCtx, bool useRdma);
    virtual HcclResult GetNetDevCtx(HcclNetDevCtx& netDevCtx, bool useRdma);
    void SetTCAndSL(u32 trafficClass, u32 serviceLevel);

protected:
    HcclNetDevCtx netDevRdmaCtx_{};
    HcclNetDevCtx netDevIpcCtx_{};
    HcclDispatcher dispatcher_{};
    HcclRankLinkInfo localRankInfo_{};
    HcclRankLinkInfo localRankVnicInfo_{};
    const RankTable_t* rankTable_{};
    std::unique_ptr<HcclSocketManager>& socketManager_;
    std::unique_ptr<NotifyPool>& notifyPool_;
    u32 trafficClass_;
    u32 serviceLevel_;

    std::string identifier_;
    bool isStandardCard_{false};
    std::unordered_set<u32> enableP2PRankIds_;
};
} // namespace hccl

#endif
