/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_REMOTE_RMA_BUFFER_MANAGER_H
#define HCCLV2_REMOTE_RMA_BUFFER_MANAGER_H

#include <string>
#include <memory>
#include <unordered_map>

#include "remote_rma_buffer.h"
#include "rma_conn_manager.h"
#include "buffer_type.h"

namespace Hccl {
using namespace std;
class CommunicatorImpl;
class RemoteRmaBufManager {
public:
    explicit RemoteRmaBufManager(const CommunicatorImpl& communicator);

    RemoteRmaBuffer* GetRemoteRmaBuffer(const string& opTag, const LinkData& linkData, BufferType bufType);

    unique_ptr<RemoteRmaBuffer> Create(const LinkData& linkData) const;

    void
    Bind(unique_ptr<RemoteRmaBuffer> remoteRmaBuf, const string& opTag, const LinkData& linkData, BufferType bufType);

private:
    CommunicatorImpl* comm;

    // tag -> LinkData -> BufType -> RemoteRmaBuffer
    std::unordered_map<
        std::string, std::unordered_map<
                         LinkData, std::unordered_map<BufferType, std::unique_ptr<RemoteRmaBuffer>, EnumClassHash>,
                         hash<Hccl::LinkData>>>
        remoteBufMap;
};

} // namespace Hccl
#endif
