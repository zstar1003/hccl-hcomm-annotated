/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_LOCAL_RMA_BUFFER_MANAGER_H
#define HCCLV2_LOCAL_RMA_BUFFER_MANAGER_H

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include "virtual_topo.h"
#include "dev_buffer.h"
#include "local_rma_buffer.h"
#include "buffer_type.h"

namespace Hccl {
// Keep fake registration isolated from the real CCL buffer registered under commId.
inline string GetAivUrmaBufferTag(const string& commId) { return commId + "_aiv_urma"; }

class CommunicatorImpl;
class LocalRmaBufManager {
public:
    explicit LocalRmaBufManager(const CommunicatorImpl& communicator);

    ~LocalRmaBufManager();

    LocalRmaBuffer*
    Reg(const string& opTag, BufferType bufferType, std::shared_ptr<Buffer> buffer, const PortData& portData,
        LinkProtocol linkProtocol);

    HcclResult Dereg(const string& opTag);

    LocalRmaBuffer* Get(const string& opTag, const PortData& portData, BufferType bufferType);

    LocalRmaBuffer* Get(const PortData& portData);

    void Destroy();

private:
    CommunicatorImpl* comm;

    bool IsExist(const string& opTag, const PortData& portData, BufferType bufferType);

    unordered_map<
        string,
        unordered_map<
            PortData, unordered_map<BufferType, unique_ptr<LocalRmaBuffer>, EnumClassHash>, hash<Hccl::PortData>>>
        bufs;

    unordered_map<PortData, unique_ptr<LocalRmaBuffer>, hash<Hccl::PortData>> ccuBufs;
};

} // namespace Hccl
#endif
