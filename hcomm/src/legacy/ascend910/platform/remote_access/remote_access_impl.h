/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_ACCESS_IMPL_H
#define REMOTE_ACCESS_IMPL_H

#include <vector>

#include "remote_access.h"
#include "hccl/base.h"
#include "transport_remote_access.h"
#include "comm_remote_access.h"

namespace hccl {
class RemoteAccessImpl {
public:
    explicit RemoteAccessImpl();
    virtual ~RemoteAccessImpl();
    HcclResult Init(u32 rank, const std::vector<MemRegisterAddr>& addrInfos, const RmaRankTable& rankTable);
    HcclResult RemoteRead(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream);
    HcclResult RemoteWrite(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream);

private:
    void ParseRemoteAccessAddrInfo(
        const std::vector<HcomRemoteAccessAddrInfo>& addrInfos,
        std::map<u32, std::vector<HcomRemoteAccessAddrInfo>>& addrInfoMap);
    HcclResult IsInSamePlane(const u32 userRank, const std::vector<HcomRemoteAccessAddrInfo>& addrInfos);
    std::unique_ptr<CommRemoteAccess> comm_;
    u32 userRank_;
    u32 userRankNum_;
    u32 serverNum_;
    u32 rankNumPerServer_; // 默认约束：每个server的rank数必须相同
};
} // namespace hccl

#endif // REMOTE_ACCESS_IMPL_H
