/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_ACCESS_H
#define REMOTE_ACCESS_H

#include <memory>
#include <vector>
#include "hccl/base.h"
#include "hccl_common.h"

namespace hccl {
class RemoteAccessImpl;

// RemoteAccess的ranktable格式
using RmaRankTable = struct tagRmaRankTable {
    u32 serverNum{0};                                  // 集群内服务器总数
    u32 rankNum{0};                                    // 通信域内的rank总数
    s32 devicePhyId{0};                                // 服务器内device唯一标识
    std::vector<std::vector<HcclIpAddress>> deviceIps; // 所有rank的device 对应的网卡ip
};

class RemoteAccess {
public:
    explicit RemoteAccess();
    virtual ~RemoteAccess();
    HcclResult Init(u32 rank, const std::vector<MemRegisterAddr>& addrInfos, const RmaRankTable& rankTable);
    HcclResult RemoteRead(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream);
    HcclResult RemoteWrite(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream);

private:
    std::unique_ptr<RemoteAccessImpl> impl_;
};
} // namespace hccl

#endif // REMOTE_ACCESS_H
