/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_OPBASE_STREAM_MANAGER_H
#define HCCLV2_OPBASE_STREAM_MANAGER_H

#include <vector>
#include <memory>
#include <unordered_map>
#include "hccl/base.h"
#include "stream/stream.h"

namespace Hccl {
class CommunicatorImpl;

class OpbaseStreamManager {
public:
    explicit OpbaseStreamManager(CommunicatorImpl* comm) : comm(comm)
    {
        CHECK_NULLPTR(comm, "[OpbaseStreamManager]comm is nullptr!");
    }
    ~OpbaseStreamManager();

    void RegisterMaster(std::unique_ptr<Stream> stream);

    void Clear();

    Stream* GetMaster() const { return master.get(); }

    Stream* GetOrCreateSlave();

    void ResetIndex(u32 index) { slaveIndex = index; }

    u32 GetSlaveIndex() const { return slaveIndex; }

    Stream* GetSlave(u32 index) const;

private:
    void ReplaceMaster(std::unique_ptr<Stream> stream);

    CommunicatorImpl* comm{nullptr};
    std::unique_ptr<Stream> master{nullptr};
    std::vector<std::unique_ptr<Stream>> slaves;
    u32 slaveIndex{0};
};

} // namespace Hccl

#endif // HCCLV2_OPBASE_STREAM_MANAGER_H
